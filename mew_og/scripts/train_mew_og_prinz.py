#!/usr/bin/env python
"""Train MEW-OG on the Prinz potential."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mew_og.config import load_config
from mew_og.guidance.augmenter import Augmenter
from mew_og.guidance.mew_og_model import MewOGModel
from mew_og.guidance.scaling import ExponentialScaling
from mew_og.io.checkpoints import load_checkpoint
from mew_og.io.hdf5 import read_hdf5
from mew_og.models.beta_schedule import LinearBetaScheduler
from mew_og.models.oggm_compat import load_oggm_score_model
from mew_og.models.score_network import ScoreBasedDDPM
from mew_og.observables.gmm import load_gmm_params
from mew_og.training.experiments import generate_experiments
from mew_og.training.train_mew_og import MewOGTrainer
from mew_og.utils.paths import get_project_root
from mew_og.utils.seed import set_seed


PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "mew_og/config/prinz_mew_og_oggm_pretrained.json"


def main():
    parser = argparse.ArgumentParser(
        description="Train MEW-OG on Prinz samples."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="JSON config file.",
    )
    parser.add_argument(
        "--biased_model",
        type=str,
        default=None,
        help="Biased DDPM checkpoint.",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        choices=["auto", "mew_og", "oggm"],
        default=None,
        help="Checkpoint format.",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default=None,
        help="Lambda .npy file.",
    )
    parser.add_argument(
        "--reweighting_results",
        type=str,
        default=None,
        help="Reweighting HDF5 file.",
    )
    parser.add_argument(
        "--observable_params",
        type=str,
        default=None,
        help="Observable parameter file.",
    )
    parser.add_argument(
        "--gt_data",
        type=str,
        default=None,
        help="Ground-truth samples HDF5.",
    )
    parser.add_argument(
        "--gt_dataset",
        type=str,
        default=None,
        help="Ground-truth dataset path.",
    )
    parser.add_argument(
        "--biased_data",
        type=str,
        default=None,
        help="Biased samples HDF5.",
    )
    parser.add_argument(
        "--biased_dataset",
        type=str,
        default="trajectory",
        help="Biased dataset path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--n_calls",
        type=int,
        default=None,
        help="Optimization iterations.",
    )
    parser.add_argument(
        "--optimizer_threshold",
        type=float,
        default=None,
        help="Optimizer loss threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run id.",
    )

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    set_seed(args.seed)

    config = load_config(args.config)

    device = args.device or config["device"]
    config["device"] = device
    biased_model = _path_from_cli_or_config(args.biased_model, config, "biased_model")
    lambdas_path = _optional_path_from_cli_or_config(args.lambdas, config, "lambdas")
    reweighting_results = _optional_path_from_cli_or_config(
        args.reweighting_results,
        config,
        "reweighting_results",
        "reweighting-results",
    )
    observable_params = _path_from_cli_or_config(
        args.observable_params,
        config,
        "observable_params",
    )
    gt_data = _optional_path_from_cli_or_config(args.gt_data, config, "gt_data")
    gt_dataset = args.gt_dataset or config.get("gt_dataset", "trajectory")
    output_dir = Path(_path_from_cli_or_config(args.output_dir, config, "output_dir"))

    optimizer_config = config["optimizer_kwargs"]
    n_calls = args.n_calls if args.n_calls is not None else optimizer_config["n_calls"]
    optimizer_kwargs = dict(optimizer_config)
    optimizer_kwargs.pop("n_calls", None)
    if args.optimizer_threshold is not None:
        optimizer_kwargs["threshold"] = args.optimizer_threshold
    config["optimizer_kwargs"]["n_calls"] = n_calls
    config["optimizer_kwargs"].update(optimizer_kwargs)
    model_format = args.model_format or config["model_format"]
    args.run = args.run or str(config.get("run", "0"))

    output_dir.mkdir(parents=True, exist_ok=True)

    model, extra = _load_biased_model(
        biased_model,
        config=config,
        device=device,
        model_format=model_format,
    )
    model.eval()

    if "config" in extra:
        for key in ["model_kwargs", "beta_scheduler_kwargs"]:
            if key in extra["config"] and key not in config:
                config[key] = extra["config"][key]

    if lambdas_path is not None:
        lambdas = torch.from_numpy(np.load(lambdas_path)).float()
    else:
        reweighting_results = _require_value(
            reweighting_results,
            "Provide --lambdas or --reweighting_results/config.reweighting_results.",
        )
        lambdas = read_hdf5(reweighting_results, "lambdas", group_name="0")

    obs_fn = load_gmm_params(observable_params)
    observable_functions = [obs_fn]

    if reweighting_results is not None:
        samples_biased = _load_trajectory(reweighting_results, "0/biased-trajectory")
        samples_gt = _load_trajectory(reweighting_results, "0/ground-truth-trajectory")
    else:
        biased_data = _require_value(args.biased_data, "Provide --biased_data.")
        gt_data = _require_value(gt_data, "Provide --gt_data or config.gt_data.")
        samples_biased = _load_trajectory(biased_data, args.biased_dataset)
        samples_gt = _load_trajectory(gt_data, gt_dataset)

    plot_samples_gt = samples_gt
    if reweighting_results is not None and gt_data is not None:
        plot_samples_gt = _load_trajectory(gt_data, gt_dataset)

    experiments = generate_experiments(
        observable_functions, samples_biased, samples_gt
    )

    beta_kwargs = _normalize_beta_kwargs(config["beta_scheduler_kwargs"])
    beta_fn = LinearBetaScheduler(device=device, **beta_kwargs)

    scaling_functions = [ExponentialScaling() for _ in experiments]
    augmenter = Augmenter(
        experimental_data=experiments,
        lambdas=lambdas,
        scaling_function=scaling_functions,
        device=device,
    )

    mew_og_model = MewOGModel(
        base_model=model,
        augmenter=augmenter,
        beta_fn=beta_fn,
        config=config,
        device=device,
    )

    trainer = MewOGTrainer(
        model=mew_og_model,
        config=config,
        output_dir=output_dir,
        ground_truth_trajectory=samples_gt,
        biased_trajectory=samples_biased,
        device=device,
    )

    result = trainer.train(
        kind=config["optimization_kind"],
        n_calls=n_calls,
        seed=args.seed,
        **optimizer_kwargs,
    )

    print(f"Best loss: {result['best_loss']:.6f}; params: {result['best_params']}")

    print("Evaluating")
    eval_results = trainer.evaluate(
        n_samples=config["n_eval_samples"],
        save_results=True,
        group_name=args.run,
    )
    plot_paths = _plot_density_observable_comparison(
        samples_biased=samples_biased,
        samples_gt=plot_samples_gt,
        eval_results=eval_results,
        augmenter=augmenter,
        output_dir=output_dir,
        run_name=args.run,
        device=device,
        seed=args.seed,
    )
    for plot_path in plot_paths:
        print(f"Saved {plot_path}")

    config["seed"] = args.seed
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved {output_dir}")


def _load_biased_model(
    model_path: str,
    config: dict,
    device: str,
    model_format: str = "auto",
) -> tuple:
    path = Path(model_path)
    use_oggm = model_format == "oggm" or (
        model_format == "auto" and path.name.endswith(".pth.tar")
    )

    if use_oggm:
        model = load_oggm_score_model(
            path,
            config=config.get("model_kwargs"),
            device=device,
        )
        return model, {"config": {"model_kwargs": model.config}}

    model = ScoreBasedDDPM.from_config(config["model_kwargs"])
    model, _, _, extra = load_checkpoint(path, model=model, device=device)
    return model, extra


def _normalize_beta_kwargs(beta_kwargs: dict) -> dict:
    beta_kwargs = dict(beta_kwargs)
    beta_kwargs.pop("device", None)
    if "a" in beta_kwargs and "beta_min" not in beta_kwargs:
        beta_kwargs["beta_min"] = beta_kwargs.pop("a")
    if "b" in beta_kwargs and "beta_max" not in beta_kwargs:
        beta_kwargs["beta_max"] = beta_kwargs.pop("b")
    return beta_kwargs


def _path_from_cli_or_config(
    arg_value: str | None,
    config: dict,
    *keys: str,
) -> str:
    """Return a CLI path or required config path, resolved from the project root."""
    value = arg_value if arg_value is not None else _first_config_value(config, *keys)
    return str(_project_path(value))


def _optional_path_from_cli_or_config(
    arg_value: str | None,
    config: dict,
    *keys: str,
) -> str | None:
    """Return an optional CLI/config path, resolved from the project root."""
    value = arg_value if arg_value is not None else _first_config_value(
        config,
        *keys,
        required=False,
    )
    return None if value is None else str(_project_path(value))


def _first_config_value(
    config: dict,
    *keys: str,
    required: bool = True,
):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]

    if required:
        formatted_keys = ", ".join(keys)
        raise KeyError(f"Missing required config value: {formatted_keys}")
    return None


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_value(value, message: str):
    if value is None:
        raise ValueError(message)
    return value


def _load_trajectory(file_path: str, dataset_path: str) -> torch.Tensor:
    dataset_parts = dataset_path.split("/")
    if len(dataset_parts) > 1:
        group_name = "/".join(dataset_parts[:-1])
        dataset_name = dataset_parts[-1]
    else:
        group_name = None
        dataset_name = dataset_path

    trajectory = read_hdf5(file_path, dataset_name, group_name=group_name)
    if trajectory.dim() == 1:
        return trajectory.unsqueeze(-1)
    return trajectory


def _plot_density_observable_comparison(
    samples_biased: torch.Tensor,
    samples_gt: torch.Tensor,
    eval_results: dict,
    augmenter: Augmenter,
    output_dir: Path,
    run_name: str,
    device: str,
    seed: int | None = None,
) -> list[Path]:
    """Plot original/guided/ground-truth densities and observable predictions."""
    if seed is not None:
        set_seed(seed)

    guided_samples = eval_results["samples"]
    plot_data = {
        "Original": _to_numpy_1d(samples_biased),
        "Guided": _to_numpy_1d(guided_samples),
        "Ground truth": _to_numpy_1d(samples_gt),
    }
    predictions = {
        name: _predict_observable_mean(augmenter, values, device)
        for name, values in plot_data.items()
    }

    fig, (ax_density, ax_obs) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )

    all_values = np.concatenate(list(plot_data.values()))
    bins = np.linspace(np.nanmin(all_values), np.nanmax(all_values), 80)
    colors = {
        "Original": "tab:blue",
        "Guided": "tab:orange",
        "Ground truth": "tab:green",
    }
    for name, values in plot_data.items():
        ax_density.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=colors[name],
            label=name,
        )

    ax_density.set_title("Sample Densities")
    ax_density.set_xlabel("x")
    ax_density.set_ylabel("Density")
    ax_density.legend(frameon=False)

    labels = list(predictions.keys())
    prediction_values = [predictions[label] for label in labels]
    bars = ax_obs.bar(
        labels,
        prediction_values,
        color=[colors[label] for label in labels],
        alpha=0.8,
    )
    ax_obs.set_title("Observable Predictions")
    ax_obs.set_ylabel("Observable mean")
    ax_obs.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, prediction_values):
        ax_obs.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("MEW-OG Density and Observable Comparison")
    fig.tight_layout()

    base_path = output_dir / f"density-observable-comparison-{run_name}"
    output_paths = [base_path.with_suffix(".pdf"), base_path.with_suffix(".png")]
    for path in output_paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def _to_numpy_1d(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _predict_observable_mean(
    augmenter: Augmenter,
    values: np.ndarray,
    device: str,
) -> float:
    samples = torch.from_numpy(values).float().view(-1, 1).to(device)
    with torch.no_grad():
        obs_per_sample = augmenter.transform(samples)
        prediction = augmenter.predict_expectations(obs_per_sample)
    return float(prediction.squeeze().detach().cpu())


if __name__ == "__main__":
    main()

