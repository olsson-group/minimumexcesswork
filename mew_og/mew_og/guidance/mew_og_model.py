"""Main MEW-OG (Minimum-Excess-Work Observable Guidance) model."""

from typing import Optional, Union

import torch
from torch import nn

from mew_og.guidance.augmenter import Augmenter
from mew_og.samplers.vp_sde import VPSDESampler


class MewOGModel(nn.Module):
    """
    Minimum-Excess-Work Observable Guidance Model.

    This model wraps a trained DDPM with an augmenter that guides sampling
    toward matching experimental observables.

    Parameters
    ----------
    base_model : torch.nn.Module
        Trained score network (DDPM).
    augmenter : Augmenter
        Guidance augmenter with lambdas and scaling function.
    beta_fn : callable
        Beta schedule for the diffusion process.
    config : dict, optional
        Configuration dictionary.
    device : str or torch.device
        Device for computations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        augmenter: Augmenter,
        beta_fn,
        config: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.base_model = base_model
        self.augmenter = augmenter
        self.beta_fn = beta_fn
        self.config = config or {}
        self.device = device

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Set up sampler
        self.sampler = VPSDESampler(
            score_network=self.base_model,
            beta_fn=self.beta_fn,
            n_atoms=self.config.get("n_atoms", 1),
            n_dim=self.config.get("n_dim", 1),
            dt=self.config.get("dt", 0.01),
            device=device,
            probability_flow=self.config.get("probability_flow", False),
        )
        self.sampler.augmenter = self.augmenter

    def forward(
        self,
        n_samples: int = 1000,
        return_all_samples: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate guided samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        return_all_samples : bool
            If True, return samples at all time steps.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        return self.sampler(
            n_samples=n_samples,
            return_all_samples=return_all_samples,
            seed=seed,
        )

    def sample(
        self,
        n_samples: int = 1000,
        return_all_samples: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(n_samples, return_all_samples, seed)

    def predict_observables(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute observables for given samples.

        Parameters
        ----------
        samples : torch.Tensor
            Samples.

        Returns
        -------
        torch.Tensor
            Mean observable values.
        """
        obs_per_sample = self.augmenter.predict_observables_per_sample(samples)
        return self.augmenter.predict_expectations(obs_per_sample)

    @property
    def scaling_function(self):
        """Get scaling function(s) from augmenter."""
        return self.augmenter.scaling_function

    def get_scaling_params(self) -> dict:
        """
        Get all scaling function parameters.

        Returns
        -------
        dict
            Dictionary of parameter names to values.
        """
        params = {}
        for i, sf in enumerate(self.augmenter.scaling_function):
            for name, param in sf.named_parameters():
                params[f"sf{i}_{name}"] = param.detach().clone()
        return params

    def set_scaling_params(self, params: dict) -> None:
        """
        Set scaling function parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names to values.
        """
        with torch.no_grad():
            for i, sf in enumerate(self.augmenter.scaling_function):
                for name, param in sf.named_parameters():
                    key = f"sf{i}_{name}"
                    if key in params:
                        param.copy_(params[key])

