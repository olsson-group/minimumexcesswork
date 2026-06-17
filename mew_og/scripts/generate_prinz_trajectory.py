#!/usr/bin/env python
"""Generate Prinz potential samples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mew_og.data.prinz import (
    prinz_potential,
    generate_deeptime_prinz_trajectory,
    bias_trajectory,
)
from mew_og.io.hdf5 import write_hdf5
from mew_og.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate Prinz potential samples."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100000,
        help="Number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--biased",
        action="store_true",
        help="Apply bias.",
    )
    parser.add_argument(
        "--bias_coefficient",
        type=float,
        default=-4.0,
        help="Bias coefficient.",
    )
    parser.add_argument(
        "--save_biased",
        action="store_true",
        help="Also save biased samples.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="trajectory",
        help="HDF5 dataset name.",
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_samples} Prinz samples")

    trajectory = generate_deeptime_prinz_trajectory(
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print(_summary("Trajectory", trajectory))

    if args.biased:
        print(f"Applying bias: {args.bias_coefficient}")
        trajectory = bias_trajectory(
            trajectory, coefficient=args.bias_coefficient, seed=args.seed
        )
        print(_summary("Biased", trajectory))

    data_dict = {args.dataset_name: trajectory}

    trajectory_biased = None
    if args.save_biased and not args.biased:
        print(f"Generating biased samples: {args.bias_coefficient}")
        trajectory_biased = bias_trajectory(
            trajectory, coefficient=args.bias_coefficient, seed=args.seed
        )
        data_dict[f"{args.dataset_name}_biased"] = trajectory_biased
        print(_summary("Biased", trajectory_biased))

    write_hdf5(output_path, data_dict)
    print(f"Saved {output_path}")

    if args.plot:
        set_seed(args.seed)
        plot_output = args.plot_output
        if plot_output is None:
            plot_output = output_path.parent / "trajectory.pdf"

        n_plots = 2 if (args.save_biased and trajectory_biased is not None) else 1
        fig, axes = plt.subplots(1, n_plots + 1, figsize=(5 * (n_plots + 1), 4))

        if n_plots == 1:
            axes = [axes[0], axes[1]]

        ax = axes[0]
        x = torch.linspace(-1.5, 1.5, 500)
        V = prinz_potential(x)
        ax.plot(x.numpy(), V.numpy(), "k-", lw=2)
        ax.set_xlabel("x")
        ax.set_ylabel("V(x)")
        ax.set_title("Prinz Potential")
        ax.set_ylim(0, 5)

        ax = axes[1]
        ax.hist(
            trajectory.numpy(),
            bins=100,
            density=True,
            alpha=0.7,
            color="steelblue",
            label="Ground truth" if args.save_biased else "Trajectory",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_title("Sample Distribution")
        ax.set_xlim(-1.5, 1.5)
        x_density = torch.linspace(-1.5, 1.5, 1000)
        equilibrium_density = torch.exp(-prinz_potential(x_density))
        equilibrium_density = equilibrium_density / torch.trapz(
            equilibrium_density, x_density
        )
        ax.plot(
            x_density.numpy(),
            equilibrium_density.numpy(),
            "k--",
            linewidth=1.5,
            label="Equilibrium",
        )
        ax.legend()

        if args.save_biased and trajectory_biased is not None:
            ax.hist(
                trajectory_biased.numpy(),
                bins=100,
                density=True,
                alpha=0.5,
                color="coral",
                label="Biased",
            )
            ax.legend()

        if n_plots == 2:
            ax = axes[2]
            ax.hist(
                trajectory_biased.numpy(),
                bins=100,
                density=True,
                alpha=0.7,
                color="coral",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            ax.set_title(f"Biased Distribution (coef={args.bias_coefficient})")
            ax.set_xlim(-1.5, 1.5)

        plt.tight_layout()
        plt.savefig(plot_output, dpi=150, bbox_inches="tight")
        print(f"Saved {plot_output}")


def _summary(label: str, values: torch.Tensor) -> str:
    return (
        f"{label}: mean={values.mean().item():.4f}, "
        f"std={values.std().item():.4f}, "
        f"min={values.min().item():.4f}, "
        f"max={values.max().item():.4f}"
    )


if __name__ == "__main__":
    main()


