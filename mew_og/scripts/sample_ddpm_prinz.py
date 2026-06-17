#!/usr/bin/env python
"""Sample the pretrained biased Prinz DDPM."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mew_og.config import load_config
from mew_og.io.hdf5 import write_hdf5
from mew_og.models.beta_schedule import LinearBetaScheduler
from mew_og.models.oggm_compat import load_oggm_score_model
from mew_og.samplers.vp_sde import VPSDESampler
from mew_og.utils.paths import get_project_root
from mew_og.utils.seed import set_seed


PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "mew_og/config/prinz_mew_og_oggm_pretrained.json"
DEFAULT_MODEL = (
    PROJECT_ROOT / "trained_models/toy/prinz-potential/biased-model/model.pth.tar"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/biased_ddpm_samples.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the biased Prinz DDPM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Biased DDPM checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="JSON config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output HDF5 file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="trajectory",
        help="Output dataset name.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampler step size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--probability_flow",
        action="store_true",
        help="Use probability-flow ODE.",
    )
    parser.add_argument(
        "--return_all_samples",
        action="store_true",
        help="Save all reverse-SDE steps.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save plot.",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Plot path.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    config = load_config(args.config) if args.config is not None else {}
    model_path = args.model or config.get("biased_model", str(DEFAULT_MODEL))
    device = args.device or config.get("device", "cpu")
    dt = args.dt or config.get("dt", 0.01)

    print(f"Loading model: {model_path}")
    model = load_oggm_score_model(
        model_path,
        config=config.get("model_kwargs", {}),
        device=device,
    )
    model.eval()

    beta_kwargs = _normalize_beta_kwargs(
        config.get("beta_scheduler_kwargs", {"beta_min": 0.1, "beta_max": 20.0})
    )
    beta_fn = LinearBetaScheduler(device=device, **beta_kwargs)
    sampler = VPSDESampler(
        score_network=model,
        beta_fn=beta_fn,
        n_atoms=config.get("n_atoms", 1),
        n_dim=config.get("n_dim", 1),
        dt=dt,
        device=device,
        probability_flow=args.probability_flow or config.get("probability_flow", False),
    )

    print(f"Sampling {args.n_samples}")
    samples = _sample_in_batches(
        sampler=sampler,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        return_all_samples=args.return_all_samples,
    )

    final_samples = _final_samples(samples, args.return_all_samples)
    trajectory = final_samples.detach().cpu().numpy().reshape(-1)

    output_path = Path(args.output)
    data = {args.dataset_name: trajectory}
    if args.return_all_samples:
        data["all_samples"] = samples.detach().cpu()
    write_hdf5(output_path, data, mode="w")

    print(
        f"Samples: mean={trajectory.mean():.4f}, std={trajectory.std():.4f}, "
        f"min={trajectory.min():.4f}, max={trajectory.max():.4f}"
    )
    print(f"Saved {output_path}")

    _save_run_config(args, config, output_path, model_path)

    if args.plot:
        set_seed(args.seed)
        plot_output = Path(args.plot_output) if args.plot_output else None
        plot_path = _plot_samples(trajectory, output_path, plot_output)
        print(f"Saved {plot_path}")


def _sample_in_batches(
    sampler: VPSDESampler,
    n_samples: int,
    batch_size: int,
    return_all_samples: bool,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    batches = []
    n_remaining = n_samples
    while n_remaining > 0:
        current_batch = min(batch_size, n_remaining)
        with torch.no_grad():
            batches.append(
                sampler(
                    n_samples=current_batch,
                    return_all_samples=return_all_samples,
                ).detach()
            )
        n_remaining -= current_batch

    if return_all_samples:
        return torch.cat(batches, dim=1)
    return torch.cat(batches, dim=0)


def _final_samples(samples: torch.Tensor, return_all_samples: bool) -> torch.Tensor:
    if return_all_samples:
        return samples[-1]
    return samples


def _normalize_beta_kwargs(beta_kwargs: dict) -> dict:
    beta_kwargs = dict(beta_kwargs)
    beta_kwargs.pop("device", None)
    if "a" in beta_kwargs and "beta_min" not in beta_kwargs:
        beta_kwargs["beta_min"] = beta_kwargs.pop("a")
    if "b" in beta_kwargs and "beta_max" not in beta_kwargs:
        beta_kwargs["beta_max"] = beta_kwargs.pop("b")
    return beta_kwargs


def _save_run_config(
    args: argparse.Namespace,
    config: dict,
    output_path: Path,
    model_path: str,
) -> None:
    run_config = dict(config)
    run_config.update(
        {
            "model": args.model,
            "resolved_model": model_path,
            "output": str(output_path),
            "dataset_name": args.dataset_name,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "dt": args.dt,
            "device": args.device,
            "seed": args.seed,
            "probability_flow": args.probability_flow,
        }
    )
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(run_config, f, indent=2)


def _plot_samples(
    trajectory: np.ndarray,
    output_path: Path,
    plot_output: Path | None,
) -> Path:
    if plot_output is None:
        plot_output = output_path.parent / "biased_ddpm_samples.pdf"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trajectory, bins=100, density=True, alpha=0.75, color="steelblue")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Samples from Biased Prinz DDPM")
    fig.tight_layout()
    fig.savefig(plot_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_output


if __name__ == "__main__":
    main()
