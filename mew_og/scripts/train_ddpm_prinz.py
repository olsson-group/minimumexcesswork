#!/usr/bin/env python
"""Train a Prinz DDPM."""

import argparse
import json
from pathlib import Path

import torch

from mew_og.config import load_config
from mew_og.data.dataloader import create_data_loader
from mew_og.data.prinz import bias_trajectory
from mew_og.io.hdf5 import read_hdf5
from mew_og.training.train_ddpm import DDPMTrainer
from mew_og.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Train a Prinz DDPM."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file.",
    )
    parser.add_argument(
        "--data",
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
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=None,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
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
    parser.add_argument(
        "--gt_data",
        type=str,
        default=None,
        help="Ground-truth trajectory for evaluation.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = {
            "n_epochs": 100,
            "batch_size": 256,
            "lr": 1e-4,
            "eval_frequency": 20,
            "device": "cpu",
            "model_kwargs": {
                "n_atoms": 1,
                "dim": 1,
                "time_embedding_dim": 3,
                "hidden_dim": 64,
                "n_layers": 3,
            },
            "beta_scheduler_kwargs": {
                "beta_min": 0.1,
                "beta_max": 20.0,
            },
        }

    if args.n_epochs is not None:
        config["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.device is not None:
        config["device"] = args.device

    device = config.get("device", "cpu")

    dataset_parts = args.dataset.split("/")
    if len(dataset_parts) > 1:
        group_name = "/".join(dataset_parts[:-1])
        dataset_name = dataset_parts[-1]
    else:
        group_name = None
        dataset_name = args.dataset

    data = read_hdf5(args.data, dataset_name, group_name=group_name)
    print(f"Loaded {len(data)} samples from {args.data}")

    if args.biased or config.get("biased", False):
        bias_coef = args.bias_coefficient or config.get("bias_coefficient", -4.0)
        print(f"Applying bias: {bias_coef}")
        data = bias_trajectory(data, coefficient=bias_coef, seed=args.seed)

    data = data.unsqueeze(-1).unsqueeze(-1)

    data_loader = create_data_loader(
        data,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
    )

    gt_trajectory = None
    if args.gt_data is not None:
        gt_trajectory = read_hdf5(args.gt_data, dataset_name, group_name=group_name)

    trainer = DDPMTrainer.from_config(
        config=config,
        data_loader=data_loader,
        output_dir=args.output_dir,
        device=device,
        ground_truth_trajectory=gt_trajectory,
    )

    print("Training DDPM")
    trainer.train()

    output_dir = Path(args.output_dir)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()

