"""Experiment containers for observable guidance."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch


@dataclass
class StaticExperiment:
    """
    Container for experimental observable data.

    This class holds the target observable value from experiments,
    along with the function to compute observables from samples.

    Attributes
    ----------
    observables_exp : torch.Tensor
        Experimental (target) observable value(s).
    observables_exp_uncertainty : torch.Tensor
        Uncertainty in the experimental values.
    observables_function : callable
        Function that computes observable from samples.
    observables_msm : torch.Tensor, optional
        Observable value from the biased/MSM distribution.
    name : str
        Name identifier for this experiment.
    kind : str
        Type of observable ('expectation' or 'functional').
    weighted_transformation : callable, optional
        Transformation applied to weighted observables.
    lmbda : torch.Tensor, optional
        Precomputed reweighting Lagrange multiplier for this observable
        (used by the BioEmu protein benchmark).
    resid : int, optional
        Residue index this observable is associated with (protein benchmarks).
    """

    observables_exp: torch.Tensor
    observables_exp_uncertainty: torch.Tensor
    observables_function: Callable
    observables_msm: Optional[torch.Tensor] = None
    name: str = "experiment"
    kind: str = "expectation"
    weighted_transformation: Optional[Callable] = None
    lmbda: Optional[torch.Tensor] = None
    resid: Optional[int] = None

    def __post_init__(self):
        """Ensure tensors are properly formatted."""
        if not isinstance(self.observables_exp, torch.Tensor):
            self.observables_exp = torch.tensor(self.observables_exp, dtype=torch.float32)
        if not isinstance(self.observables_exp_uncertainty, torch.Tensor):
            self.observables_exp_uncertainty = torch.tensor(
                self.observables_exp_uncertainty, dtype=torch.float32
            )
        if self.observables_msm is not None and not isinstance(self.observables_msm, torch.Tensor):
            self.observables_msm = torch.tensor(self.observables_msm, dtype=torch.float32)
        if self.lmbda is not None and not isinstance(self.lmbda, torch.Tensor):
            self.lmbda = torch.tensor(self.lmbda, dtype=torch.float32)

        # Ensure at least 1D
        if self.observables_exp.dim() == 0:
            self.observables_exp = self.observables_exp.unsqueeze(0)
        if self.observables_exp_uncertainty.dim() == 0:
            self.observables_exp_uncertainty = self.observables_exp_uncertainty.unsqueeze(0)

        # Default transformation is identity
        if self.weighted_transformation is None:
            self.weighted_transformation = lambda x: x


def generate_experiments(
    observable_functions: List[Callable],
    samples_biased: torch.Tensor,
    samples_gt: torch.Tensor,
    uncertainty: float = 0.1,
) -> List[StaticExperiment]:
    """
    Generate experiments from observable functions and ground truth samples.

    This creates StaticExperiment objects where the 'experimental' values are
    computed from ground truth samples, and the biased/MSM values are computed
    from biased samples.

    Parameters
    ----------
    observable_functions : list of callable
        List of observable functions.
    samples_biased : torch.Tensor
        Samples from the biased distribution.
    samples_gt : torch.Tensor
        Samples from the ground truth distribution.
    uncertainty : float
        Uncertainty to assign to experimental values.

    Returns
    -------
    list of StaticExperiment
        List of experiment objects.
    """
    experiments = []

    for i, obs_fn in enumerate(observable_functions):
        # Compute observables
        obs_gt = obs_fn(samples_gt)
        obs_biased = obs_fn(samples_biased)

        experiment = StaticExperiment(
            observables_exp=torch.mean(obs_gt, dim=0, keepdim=True),
            observables_exp_uncertainty=torch.tensor([uncertainty]),
            observables_msm=torch.mean(obs_biased, dim=0, keepdim=True),
            observables_function=obs_fn,
            name=f"obs_{i}",
        )
        experiments.append(experiment)

    return experiments

