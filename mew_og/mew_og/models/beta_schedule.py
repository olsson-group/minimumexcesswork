"""Beta noise schedules for diffusion models."""

from abc import ABC, abstractmethod
from typing import Union

import torch


class BetaScheduler(ABC):
    """
    Abstract base class for beta (noise) schedulers.

    The beta schedule controls the amount of noise added at each diffusion step.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        """
        Initialize the scheduler.

        Parameters
        ----------
        device : str or torch.device
            Device for tensor computations.
        """
        self.device = device

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute beta(t).

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Beta values at time t.
        """
        pass

    @abstractmethod
    def integral(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the integral of beta from 0 to t.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Integral values.
        """
        pass


class LinearBetaScheduler(BetaScheduler):
    """
    Linear beta schedule: beta(t) = a + (b - a) * t

    This is a common schedule for VP-SDE diffusion models.

    Parameters
    ----------
    beta_min : float
        Minimum beta value (at t=0).
    beta_max : float
        Maximum beta value (at t=1).
    n_steps : int
        Number of discrete steps (for discrete schedule compatibility).
    device : str or torch.device
        Device for tensor computations.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        n_steps: int = 100,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(device)

        if beta_min >= beta_max:
            raise ValueError("beta_min must be less than beta_max")
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_steps = n_steps

        # Discrete schedule for compatibility
        self.discrete_betas = torch.linspace(
            beta_min / n_steps, beta_max / n_steps, n_steps, device=device
        )
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute beta(t) = beta_min + (beta_max - beta_min) * t.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Beta values.
        """
        t = t.to(self.device)
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute integral of beta from 0 to t.

        integral(beta, 0, t) = beta_min * t + 0.5 * (beta_max - beta_min) * t^2

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Integral values.
        """
        t = t.to(self.device)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha(t) = 1 - beta(t) / n_steps for discrete compatibility.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Alpha values.
        """
        timestep = (t * (self.n_steps - 1)).to(torch.int32)
        timestep = timestep.clamp(0, self.n_steps - 1)
        return self.alphas[timestep]

    def alpha_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative product of alphas up to time t.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Cumulative alpha product values.
        """
        timestep = (t * (self.n_steps - 1)).to(torch.int32)
        timestep = timestep.clamp(0, self.n_steps - 1)
        return self.alphas_cumprod[timestep]

