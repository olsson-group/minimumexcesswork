"""Experiment generators for the BioEmu protein benchmark.

Load experimental ``3J(HN-HA)`` reference data from CSV and build
:class:`mew_og.training.experiments.StaticExperiment` objects that carry the
per-residue target value, the biased/MSM value, and the precomputed
reweighting lambda.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

from mew_og.training.experiments import StaticExperiment
from mew_og.utils.paths import get_project_root


def _resolve_csv(csv_file: str, data_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve a reference CSV path against ``data_dir`` (or project-root/data)."""
    csv_path = Path(csv_file)
    if csv_path.is_absolute():
        return csv_path
    base = Path(data_dir) if data_dir is not None else get_project_root() / "data"
    return base / csv_path


def generate_homeodomain_experiments(
    observable_functions,
    csv_file: str = "homeodomain/3jhnha_reference.csv",
    data_dir: Optional[Union[str, Path]] = None,
):
    """Build StaticExperiments for the homeodomain 3J(HN-HA) benchmark."""
    reference_csv = _resolve_csv(csv_file, data_dir)
    df = pd.read_csv(reference_csv)

    # Remove residues that occur more than once
    counts = df["resid"].value_counts()
    duplicated_resids = set(counts[counts > 1].index.tolist())
    if len(duplicated_resids) > 0:
        df = df[~df["resid"].isin(duplicated_resids)].copy()

    experiments = []
    for _, row in df.iterrows():
        exp_value = torch.tensor([[row["value"]]])
        exp_uncertainty = torch.tensor([[row["error"]]])
        bioemu_value = row["bioemu"]
        resid = row["resid"]
        lmbda_value = row["lambdas"]

        exp = StaticExperiment(
            observables_exp=torch.atleast_2d(exp_value),
            observables_exp_uncertainty=torch.atleast_2d(exp_uncertainty),
            observables_msm=torch.atleast_2d(torch.tensor([[bioemu_value]])),
            observables_function=observable_functions,
            name=str(resid),
            resid=int(resid),
            lmbda=torch.tensor([lmbda_value]),
        )
        experiments.append(exp)

    return experiments


def generate_hngl_experiments(
    observable_functions,
    csv_file: str = "hngl/3jhnha.csv",
    data_dir: Optional[Union[str, Path]] = None,
):
    """
    Load HNGL experimental 3J(HN-HA) data from CSV and build StaticExperiments.

    Expects columns ``resid``, ``value``, ``error``. If ``bioemu``/``lambdas``
    columns are missing they default to the experimental value and ``0.0``.
    """
    reference_csv = _resolve_csv(csv_file, data_dir)
    df = pd.read_csv(reference_csv)

    counts = df["resid"].value_counts()
    duplicated_resids = set(counts[counts > 1].index.tolist())
    if len(duplicated_resids) > 0:
        df = df[~df["resid"].isin(duplicated_resids)].copy()

    experiments = []
    for _, row in df.iterrows():
        exp_value = torch.tensor([[row["value"]]])
        exp_uncertainty = torch.tensor([[row.get("error", 0.0)]])
        bioemu_value = row.get("bioemu", row["value"])
        resid = row["resid"]
        lmbda_value = row.get("lambdas", 0.0)

        exp = StaticExperiment(
            observables_exp=torch.atleast_2d(exp_value),
            observables_exp_uncertainty=torch.atleast_2d(exp_uncertainty),
            observables_msm=torch.atleast_2d(torch.tensor([[bioemu_value]])),
            observables_function=observable_functions,
            name=str(resid),
            resid=int(resid),
            lmbda=torch.tensor([lmbda_value]),
        )
        experiments.append(exp)

    return experiments
