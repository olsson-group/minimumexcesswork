"""Augmenter for observable guidance in diffusion sampling."""

from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

from mew_og.guidance.scaling import ScalingFunction, ExponentialScaling


class Augmenter(nn.Module):
    """
    Guidance augmenter that computes the guidance term for sampling.

    The guidance term is:
        h(x, t) = -alpha(t) * sum_f(lambda_f * grad_x f(x))

    where:
    - alpha(t) is a time-dependent scaling function
    - lambda_f are Lagrange multipliers from MaxEnt reweighting
    - f(x) are the observable functions

    Parameters
    ----------
    experimental_data : list
        List of experiment objects with observable functions.
    lambdas : torch.Tensor
        Lagrange multipliers.
    scaling_function : ScalingFunction or list of ScalingFunction
        Time-dependent scaling function(s).
    device : str or torch.device
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.
    """

    def __init__(
        self,
        experimental_data: List,
        lambdas: torch.Tensor,
        scaling_function: Optional[Union[ScalingFunction, List[ScalingFunction]]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self._experimental_data = experimental_data
        self._excess_work = torch.zeros(1, device=device)

        # Set up lambdas
        if isinstance(lambdas, np.ndarray):
            lambdas = torch.from_numpy(lambdas)
        self._lambdas = lambdas.to(dtype).to(device)

        # Set up scaling function(s)
        if scaling_function is None:
            scaling_function = [ExponentialScaling() for _ in experimental_data]
        elif not isinstance(scaling_function, list):
            scaling_function = [scaling_function]

        # Ensure we have one scaling function per observable
        if len(scaling_function) == 1 and len(experimental_data) > 1:
            # Replicate the scaling function
            sf_type = type(scaling_function[0])
            scaling_function = [sf_type() for _ in experimental_data]

        self._scaling_function = nn.ModuleList(scaling_function)

        # Set up observable function
        self._set_observable_function()

    def _set_observable_function(self) -> None:
        """Set up the combined observable function."""
        if len(set(exp.observables_function for exp in self._experimental_data)) == 1:
            self.observables_function = self._experimental_data[0].observables_function
        else:
            def combined(x):
                return torch.hstack([exp.observables_function(x) for exp in self._experimental_data])
            self.observables_function = combined

    @property
    def lambdas(self) -> torch.Tensor:
        """Get Lagrange multipliers."""
        return self._lambdas

    @lambdas.setter
    def lambdas(self, value: torch.Tensor) -> None:
        """Set Lagrange multipliers."""
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self._lambdas = value.to(self.dtype).to(self.device)

    @property
    def scaling_function(self) -> nn.ModuleList:
        """Get scaling function(s)."""
        return self._scaling_function

    @property
    def excess_work(self) -> torch.Tensor:
        """Get accumulated excess work."""
        return self._excess_work

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the guidance term h(x, t).

        Parameters
        ----------
        x : torch.Tensor
            Samples of shape (batch_size, n_atoms, n_dim).
        t : torch.Tensor
            Time value.

        Returns
        -------
        torch.Tensor
            Guidance term of shape (batch_size, n_atoms, n_dim).
        """
        batch_size = x.shape[0]
        original_shape = x.shape

        # Flatten for observable computation
        x_flat = x.view(batch_size, -1)

        # Enable gradients for x
        x_flat = x_flat.requires_grad_(True)

        # Compute observables
        observables = self.observables_function(x_flat)

        # Compute scaling factors for each observable
        alpha_t = torch.hstack([sf(t) for sf in self._scaling_function])

        # Scale observables
        scaled_observables = observables * alpha_t

        # Compute gradients of each observable w.r.t. x
        gradients = []
        for i in range(scaled_observables.shape[1]):
            grad = torch.autograd.grad(
                outputs=scaled_observables[:, i].sum(),
                inputs=x_flat,
                create_graph=False,
                retain_graph=(i < scaled_observables.shape[1] - 1),
            )[0]
            gradients.append(grad)

        # Stack gradients: (n_observables, batch_size, n_features)
        gradients = torch.stack(gradients)

        # Compute weighted sum: h = -sum_f(lambda_f * grad_f)
        # lambdas: (n_observables,), gradients: (n_observables, batch_size, n_features)
        h = -torch.sum(self._lambdas.view(-1, 1, 1) * gradients, dim=0)

        # Compute excess work for regularization
        self._excess_work = self._compute_excess_work(alpha_t, h)

        # Reshape to original shape
        return h.view(original_shape)

    def _compute_excess_work(
        self, alpha_t: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute excess work (for regularization).

        Parameters
        ----------
        alpha_t : torch.Tensor
            Scaling factors.
        h : torch.Tensor
            Guidance term.

        Returns
        -------
        torch.Tensor
            Excess work value.
        """
        return torch.sum(alpha_t**2 * torch.mean(h**2))

    def transform(
        self, x: torch.Tensor, return_experimental: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute observables per sample.

        Parameters
        ----------
        x : torch.Tensor
            Samples.
        return_experimental : bool
            If True, also return experimental values.

        Returns
        -------
        observables_per_sample : torch.Tensor
            Observable values for each sample.
        observables_exp : torch.Tensor, optional
            Experimental observable values (if return_experimental=True).
        """
        x_flat = x.view(x.shape[0], -1)
        observables_per_sample = self.observables_function(x_flat)

        if return_experimental:
            observables_exp = torch.vstack([
                exp.observables_exp for exp in self._experimental_data
            ]).T
            return observables_per_sample, observables_exp

        return observables_per_sample

    def predict_observables_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for transform()."""
        return self.transform(x)

    def predict_expectations(self, observables_per_sample: torch.Tensor) -> torch.Tensor:
        """
        Compute mean observable values.

        Parameters
        ----------
        observables_per_sample : torch.Tensor
            Observable values for each sample.

        Returns
        -------
        torch.Tensor
            Mean observable values.
        """
        return observables_per_sample.mean(dim=0, keepdim=True)

    @property
    def observables_exp_uncertainty(self) -> torch.Tensor:
        """Get experimental uncertainties."""
        return torch.vstack([
            exp.observables_exp_uncertainty for exp in self._experimental_data
        ]).T

