#!/usr/bin/env python
"""Fit a GMM observable from trajectory data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from mew_og.io.hdf5 import read_hdf5
from mew_og.observables.gmm import (
    fit_gmm,
    save_gmm_params,
    GaussianMixtureObservable,
)
from mew_og.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Fit a GMM observable."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input HDF5 file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trajectory",
        help="HDF5 dataset path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy file.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=4,
        help="Number of components.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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

    input_path = Path(args.input)
    dataset_parts = args.dataset.split("/")
    if len(dataset_parts) > 1:
        group_name = "/".join(dataset_parts[:-1])
        dataset_name = dataset_parts[-1]
    else:
        group_name = None
        dataset_name = args.dataset

    data = read_hdf5(input_path, dataset_name, group_name=group_name)
    data = data.numpy().flatten()
    print(f"Loaded {len(data)} samples from {input_path}")

    print(f"Fitting {args.n_components}-component GMM")
    gmm = fit_gmm(
        data,
        n_components=args.n_components,
        random_state=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gmm_params(gmm, output_path)
    print(f"Saved {output_path}")

    print("Components: " + _format_components(gmm))

    if args.plot:
        plot_path = _plot_gmm(data, gmm, output_path, args.plot_output, args.seed)
        print(f"Saved {plot_path}")


def _format_components(gmm) -> str:
    parts = []
    for i, (mean, var, weight) in enumerate(
        zip(gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)
    ):
        parts.append(f"{i}: mean={mean:.3f}, var={var:.3f}, weight={weight:.3f}")
    return "; ".join(parts)


def _plot_gmm(
    data: np.ndarray,
    gmm,
    output_path: Path,
    plot_output: str | None,
    seed: int,
) -> Path:
    set_seed(seed)
    plot_path = (
        Path(plot_output) if plot_output is not None else output_path.parent / "gmm_fit.pdf"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(data, bins=50, density=True, alpha=0.6, color="steelblue", label="Data")
    x = np.linspace(data.min(), data.max(), 1000)
    for i, (mean, var, weight) in enumerate(
        zip(gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)
    ):
        ax.plot(
            x,
            weight * norm.pdf(x, mean, np.sqrt(var)),
            linestyle="--",
            alpha=0.7,
            label=f"Component {i} (mu={mean:.2f})",
        )
    ax.plot(
        x,
        np.exp(gmm.score_samples(x.reshape(-1, 1))),
        color="red",
        lw=2,
        label="GMM",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("GMM Fit")
    ax.legend(fontsize=8)

    ax = axes[1]
    obs_fn = GaussianMixtureObservable(
        np.column_stack(
            (gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)
        )
    )
    import torch
    ax.plot(x, obs_fn(torch.from_numpy(x).float()).numpy(), color="purple", lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("Observable f(x)")
    ax.set_title("Observable")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    return plot_path


if __name__ == "__main__":
    main()

