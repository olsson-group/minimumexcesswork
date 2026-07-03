#!/usr/bin/env python
"""Run the BioEmu protein benchmark (observable-guided sampling).

Ports the original hydra-based benchmark to MEW-OG conventions: an ``argparse``
CLI, a JSON config, and project-root-relative paths (no ``.env``). The DPM
denoiser is built with ``functools.partial`` from the config instead of hydra
instantiation.
"""

import argparse
import functools
import json
import logging
import os
import random
import string
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
from bioemu.seq_io import check_protein_valid, parse_sequence, write_fasta
from bioemu.utils import count_samples_in_output_dir, format_npz_samples_filename

from mew_og.benchmark.augmenter import BioEmuAugmenter
from mew_og.benchmark.experiments import (
    generate_hngl_experiments,
    generate_homeodomain_experiments,
)
from mew_og.benchmark.sampling import dpm_solver, generate_batch, save_pdb_and_xtc
from mew_og.benchmark.trainer import BioEmuTrainer
from mew_og.config import load_config
from mew_og.guidance.scaling import ExponentialScaling
from mew_og.observables.nmr import ThreeJHNHA
from mew_og.utils.paths import get_project_root

logger = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "mew_og/config/bioemu_homeodomain.json"


def main():
    parser = argparse.ArgumentParser(description="Run the BioEmu MEW-OG benchmark.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG), help="JSON config file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory holding sequences/embeddings/CSVs (default: <root>/data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory (default: <root>/out).",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    args = parser.parse_args()

    config = load_config(args.config)

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    output_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "out"

    system = config.get("system", "homeodomain")
    now_str = datetime.now().strftime("%y%m%d-%H%M")
    run_name = _random_name(f"{system}-{now_str}", 4)
    out_dir = output_root / config.get("output_root", f"tmp/{system}") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(out_dir)

    _setup_logging(out_dir / "logfile.log")
    logger.info(f"Output directory: {out_dir}")

    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    batch_size = int(config.get("batch_size", 8))
    n_training_samples = int(config.get("n_training_samples", batch_size * 10))

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=config.get("model_name"),
        ckpt_path=config.get("ckpt_path", None),
        model_config_path=config.get("model_config_path", None),
    )
    score_model = load_model(ckpt_path, model_config_path)
    cache_embeds_dir = data_dir / config["cache_embeds_dir"]

    sdes = load_sdes(
        model_config_path=model_config_path,
        cache_so3_dir=config.get("cache_so3_dir", None),
    )

    seq_arg = data_dir / config["sequence"]
    msa_file = seq_arg if str(seq_arg).endswith(".a3m") else None
    if msa_file and config.get("msa_host_url"):
        logger.warning(
            f"Ignoring msa_host_url={config['msa_host_url']!r} because "
            f"msa_file={msa_file!r} is provided."
        )

    sequence = parse_sequence(seq_arg)
    check_protein_valid(sequence)

    fasta_path = out_dir / "sequence.fasta"
    if fasta_path.exists():
        existing = parse_sequence(fasta_path)
        if existing != sequence:
            raise ValueError(
                f"{fasta_path} exists with different sequence ({existing}) vs "
                f"expected ({sequence})"
            )
    else:
        write_fasta([sequence], fasta_path)

    denoiser = _build_denoiser(config)

    logger.info(
        f"Sampling {n_training_samples} structures for sequence "
        f"length={len(sequence)}"
    )
    if batch_size <= 0:
        logger.warning("Batch size <= 0; forcing batch_size=1")
        batch_size = 1
    logger.info(f"Using batch_size={min(batch_size, n_training_samples)}")

    existing_num = count_samples_in_output_dir(out_dir)
    logger.info(f"Found {existing_num} existing samples in {out_dir}")

    scalar_fn = ThreeJHNHA(device=device)
    if system == "hngl":
        experiments = generate_hngl_experiments(scalar_fn, data_dir=data_dir)
    else:
        experiments = generate_homeodomain_experiments(scalar_fn, data_dir=data_dir)

    manual_index = config.get("manual_index", None)
    manual_lambda = config.get("manual_lambda", None)
    if manual_index is not None and manual_lambda is not None:
        try:
            idx = int(manual_index)
            experiments = [experiments[idx]]
            lambdas = torch.tensor([float(manual_lambda)], device=device)
        except Exception as e:
            logger.warning(
                f"Failed to apply manual subset (index={manual_index}, "
                f"lambda={manual_lambda}): {e}"
            )
            lambdas = torch.tensor([e.lmbda for e in experiments], device=device)
    else:
        lambdas = torch.tensor([e.lmbda for e in experiments], device=device)

    # Reuse the standard MEW-OG exponential scaling alpha(t) = a * exp(-b * t);
    # the optimizer tunes the "_b" parameter within the configured bounds.
    scaling_functions = [ExponentialScaling() for _ in experiments]

    augmenter = BioEmuAugmenter(
        experimental_data=experiments,
        scaling_function=scaling_functions,
        lambdas=lambdas,
        device=device,
        normalization=None,
    )

    if config.get("train_mew", False):
        trainer = BioEmuTrainer(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=batch_size,
            n_training_samples=n_training_samples,
            seed=0,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=config.get("msa_host_url", None),
            augmenter=augmenter,
            config=config,
            output_dir=out_dir,
            device=device,
        )
        trainer.train(
            kind=config.get("optimization_kind", "bayesian-optimization"),
            **config.get("optimizer_kwargs", {}),
        )
        trainer.evaluate(n_samples=config.get("n_evaluation_samples", 5120))
    else:
        base_seed = int.from_bytes(os.urandom(8), byteorder="big") % 1_000_000_000
        logger.info(f"Using random base seed: {base_seed}")

        for batch_idx, seed in enumerate(
            tqdm(
                range(existing_num, n_training_samples, batch_size),
                desc="Sampling batches...",
            )
        ):
            n = min(batch_size, n_training_samples - seed)
            npz_path = out_dir / format_npz_samples_filename(seed, n)
            if npz_path.exists():
                raise ValueError(
                    f"{npz_path} already exists but only {existing_num} were "
                    "generated so far."
                )

            batch_seed = base_seed + seed
            logger.info(f"Batch {batch_idx}, seed={batch_seed}")

            batch = generate_batch(
                score_model=score_model,
                sequence=sequence,
                sdes=sdes,
                batch_size=n,
                seed=batch_seed,
                denoiser=denoiser,
                cache_embeds_dir=cache_embeds_dir,
                msa_file=msa_file,
                msa_host_url=config.get("msa_host_url", None),
                augmenter=None,
            )
            arr_batch = {k: v.cpu().numpy() for k, v in batch.items()}
            np.savez(npz_path, **arr_batch, sequence=sequence)

        logger.info("Converting samples to PDB + XTC...")
        files = sorted(out_dir.glob("batch_*.npz"))
        seqs = [np.load(f)["sequence"].item() for f in files]
        if set(seqs) != {sequence}:
            raise ValueError(
                f"Mismatch in sequences: {set(seqs)} vs expected {sequence}"
            )
        pos = torch.tensor(np.concatenate([np.load(f)["pos"] for f in files]))
        ori = torch.tensor(
            np.concatenate([np.load(f)["node_orientations"] for f in files])
        )
        save_pdb_and_xtc(
            pos_nm=pos,
            node_orientations=ori,
            topology_path=out_dir / "topology.pdb",
            xtc_path=out_dir / "samples.xtc",
            sequence=sequence,
            filter_samples=config.get("filter_samples", True),
        )

    logger.info(f"All done! Samples live in {out_dir}")


def _build_denoiser(config: dict):
    """Build the DPM denoiser as a partial of :func:`dpm_solver` from config."""
    denoiser_type = config.get("denoiser_type", "dpm")
    if denoiser_type != "dpm":
        raise ValueError(
            f"Unsupported denoiser_type={denoiser_type!r}; only 'dpm' is bundled."
        )
    denoiser_cfg = config.get("denoiser", {})
    return functools.partial(
        dpm_solver,
        eps_t=float(denoiser_cfg.get("eps_t", 0.001)),
        max_t=float(denoiser_cfg.get("max_t", 0.99)),
        N=int(denoiser_cfg.get("N", 30)),
    )


def _random_name(prefix: str = "run", length: int = 4) -> str:
    rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=length))
    return f"{prefix}-{rand_str}"


def _setup_logging(logfile_path):
    logging.basicConfig(filename=logfile_path, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger(__name__).addHandler(console)


if __name__ == "__main__":
    main()
