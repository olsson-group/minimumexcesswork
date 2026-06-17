#!/usr/bin/env python
"""Fit reweighting lambdas for Prinz samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mew_og.config import load_config
from mew_og.io.checkpoints import load_checkpoint
from mew_og.io.hdf5 import read_hdf5, write_hdf5
from mew_og.models.beta_schedule import LinearBetaScheduler
from mew_og.models.score_network import ScoreBasedDDPM
from mew_og.observables.gmm import load_gmm_params
from mew_og.reweighting.maxent import MaxEntReweightingEstimator
from mew_og.samplers.vp_sde import VPSDESampler
from mew_og.training.experiments import generate_experiments
from mew_og.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Fit reweighting lambdas."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file.",
    )
    parser.add_argument(
        "--biased_model",
        type=str,
        default=None,
        help="Biased DDPM checkpoint.",
    )
    parser.add_argument(
        "--gt_model",
        type=str,
        default=None,
        help="Ground-truth DDPM checkpoint.",
    )
    parser.add_argument(
        "--biased_data",
        type=str,
        default=None,
        help="Biased samples HDF5.",
    )
    parser.add_argument(
        "--gt_data",
        type=str,
        default=None,
        help="Ground-truth samples HDF5.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trajectory",
        help="Biased dataset path.",
    )
    parser.add_argument(
        "--gt_dataset",
        type=str,
        default=None,
        help="Ground-truth dataset path.",
    )
    parser.add_argument(
        "--observable_params",
        type=str,
        default=None,
        help="Observable parameter file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Samples for fitting.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Optimization iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = {
            "n_samples": args.n_samples or 10000,
            "max_iter": args.max_iter or 50,
            "device": args.device or "cpu",
        }

    device = args.device or config["device"]
    n_samples = args.n_samples if args.n_samples is not None else config["n_samples"]
    max_iter = args.max_iter if args.max_iter is not None else config["max_iter"]
    observable_params = args.observable_params or config["observable_params"]
    output_dir = Path(args.output_dir or config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading observable: {observable_params}")
    obs_fn = load_gmm_params(observable_params)
    observable_functions = [obs_fn]

    samples_biased = None
    samples_gt = None

    dataset_parts = args.dataset.split("/")
    if len(dataset_parts) > 1:
        group_name = "/".join(dataset_parts[:-1])
        dataset_name = dataset_parts[-1]
    else:
        group_name = None
        dataset_name = args.dataset

    gt_dataset_name = args.gt_dataset
    if gt_dataset_name is None:
        if dataset_name.endswith("_biased"):
            gt_dataset_name = dataset_name[:-7]
        else:
            gt_dataset_name = dataset_name

    if args.biased_data is not None:
        print(f"Loading biased samples: {args.biased_data}:{dataset_name}")
        samples_biased = read_hdf5(args.biased_data, dataset_name, group_name=group_name)
        samples_biased = samples_biased[:n_samples].unsqueeze(-1)
    elif args.biased_model is not None:
        print(f"Sampling biased model: {args.biased_model}")
        samples_biased = _generate_samples(args.biased_model, n_samples, device, config)
    else:
        raise ValueError("Must provide either --biased_data or --biased_model")

    if args.gt_data is not None:
        print(f"Loading ground-truth samples: {args.gt_data}:{gt_dataset_name}")
        samples_gt = read_hdf5(args.gt_data, gt_dataset_name, group_name=group_name)
        samples_gt = samples_gt[:n_samples].unsqueeze(-1)
    elif args.biased_data is not None and gt_dataset_name != dataset_name:
        print(f"Loading ground-truth samples: {args.biased_data}:{gt_dataset_name}")
        samples_gt = read_hdf5(args.biased_data, gt_dataset_name, group_name=group_name)
        samples_gt = samples_gt[:n_samples].unsqueeze(-1)
    elif args.gt_model is not None:
        print(f"Sampling ground-truth model: {args.gt_model}")
        samples_gt = _generate_samples(args.gt_model, n_samples, device, config)
    else:
        raise ValueError(
            "Must provide --gt_data, --gt_model, or ground truth in the biased file"
        )

    print(f"Samples: biased={tuple(samples_biased.shape)}, gt={tuple(samples_gt.shape)}")

    experiments = generate_experiments(
        observable_functions, samples_biased, samples_gt
    )
    print("Experiments: " + _format_experiments(experiments))
    print("Fitting lambdas")
    estimator = MaxEntReweightingEstimator(
        experimental_data=experiments,
        device=device,
    )

    result = estimator.fit(
        samples_biased,
        max_iter=max_iter,
        seed=args.seed,
    )

    print(f"Lambdas: {result['lambdas'].tolist()} (error={result['error']:.6f})")

    np.save(output_dir / "lambdas.npy", result["lambdas"].numpy())
    write_hdf5(
        output_dir / "reweighting-results.h5",
        {
            "lambdas": result["lambdas"],
            "biased-trajectory": samples_biased.squeeze(),
            "ground-truth-trajectory": samples_gt.squeeze(),
            "weights": estimator.w,
        },
        group_name="0",
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved {output_dir}")


def _format_experiments(experiments: list) -> str:
    return "; ".join(
        (
            f"{i}: target={exp.observables_exp.item():.4f}, "
            f"biased={exp.observables_msm.item():.4f}"
        )
        for i, exp in enumerate(experiments)
    )


def _generate_samples(
    model_path: str,
    n_samples: int,
    device: str,
    config: dict,
) -> torch.Tensor:
    model = ScoreBasedDDPM.from_config(config.get("model_kwargs", {}))
    model, _, _, _ = load_checkpoint(model_path, model=model, device=device)
    model.eval()

    beta_kwargs = config.get("beta_scheduler_kwargs", {})
    beta_fn = LinearBetaScheduler(device=device, **beta_kwargs)

    sampler = VPSDESampler(
        score_network=model,
        beta_fn=beta_fn,
        device=device,
    )

    with torch.no_grad():
        samples = sampler(n_samples=n_samples)

    return samples.squeeze(-1)


if __name__ == "__main__":
    main()

