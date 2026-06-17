"""VP-SDE sampler for diffusion models with optional guidance."""

from typing import Callable, Optional, Union

import torch


class VPSDESampler:
    """
    Variance Preserving SDE sampler using Euler-Maruyama integration.

    This sampler generates samples from a trained score-based diffusion model
    by integrating the reverse-time SDE from t=1 to t=0.

    The reverse-time SDE is:
        dx = [-0.5 * beta(t) * x - beta(t) * (score(x,t) + h(x,t))] dt + sqrt(beta(t)) dW

    where h(x,t) is an optional guidance term.

    Parameters
    ----------
    score_network : torch.nn.Module
        Trained score network that predicts score(x, t).
    beta_fn : BetaScheduler
        Beta schedule with __call__ and integral methods.
    n_atoms : int
        Number of atoms (use 1 for 1D toy).
    n_dim : int
        Dimension per atom.
    dt : float
        Integration time step.
    device : str or torch.device
        Device for computations.
    probability_flow : bool
        If True, use ODE (probability flow) instead of SDE.
    """

    def __init__(
        self,
        score_network: torch.nn.Module,
        beta_fn: Callable,
        n_atoms: int = 1,
        n_dim: int = 1,
        dt: float = 0.01,
        device: Union[str, torch.device] = "cpu",
        probability_flow: bool = False,
    ):
        self.score_network = score_network
        self.beta_fn = beta_fn
        self.n_atoms = n_atoms
        self.n_dim = n_dim
        self.dt = dt
        self.device = device
        self.probability_flow = probability_flow

        # Optional guidance/augmenter
        self._augmenter = None
        self._excess_work = torch.tensor(0.0, device=device)

    @property
    def augmenter(self) -> Optional[Callable]:
        """Get the guidance augmenter."""
        return self._augmenter

    @augmenter.setter
    def augmenter(self, value: Optional[Callable]) -> None:
        """Set the guidance augmenter."""
        self._augmenter = value

    @property
    def excess_work(self) -> torch.Tensor:
        """Get the accumulated excess work from guidance."""
        return self._excess_work

    def __call__(
        self,
        n_samples: int = 1000,
        return_all_samples: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples by integrating the reverse-time SDE.

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
            Generated samples. Shape depends on return_all_samples:
            - If False: (n_samples, n_atoms, n_dim)
            - If True: (n_steps, n_samples, n_atoms, n_dim)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        n_steps = int(1.0 / self.dt)

        # Initialize from standard normal
        x_t = torch.randn(
            n_samples, self.n_atoms, self.n_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Storage for all samples if requested
        if return_all_samples:
            all_samples = torch.empty(
                n_steps, n_samples, self.n_atoms, self.n_dim,
                device=self.device,
            )

        # Reset excess work
        self._excess_work = torch.tensor(0.0, device=self.device)

        # Integrate from t=1 to t=0
        t = torch.tensor(1.0, device=self.device)
        for i in range(n_steps):
            x_t = self._step(x_t, t)

            if return_all_samples:
                all_samples[i] = x_t.detach()

            t = t - self.dt

        if return_all_samples:
            return all_samples
        else:
            return x_t.detach()

    def _step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform one Euler-Maruyama step of the reverse-time SDE.

        Parameters
        ----------
        x_t : torch.Tensor
            Current samples.
        t : torch.Tensor
            Current time.

        Returns
        -------
        torch.Tensor
            Updated samples.
        """
        beta_t = self.beta_fn(t)

        # Compute score
        with torch.no_grad():
            score = self.score_network(x_t, t)

        # Compute guidance on the denoised estimate, matching OGGM's sampler.
        x_0_hat = (
            x_t
            + torch.sqrt(1 - self.beta_fn.alpha(t))
            * torch.sqrt(1 - self.beta_fn.alpha_cumprod(t))
            * score
        ) / torch.sqrt(self.beta_fn.alpha(t))
        h_t = self._compute_guidance(x_0_hat.detach(), t)

        # Drift term
        drift = -0.5 * beta_t * x_t - beta_t * (score + h_t)

        # Diffusion term (0 for probability flow ODE)
        if self.probability_flow:
            diffusion = 0.0
        else:
            diffusion = torch.sqrt(beta_t)

        # Euler-Maruyama update
        noise = torch.randn_like(x_t) if not self.probability_flow else 0.0
        x_next = x_t - drift * self.dt + diffusion * noise * torch.sqrt(torch.tensor(self.dt))

        return x_next

    def _compute_guidance(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the guidance term h(x, t).

        Parameters
        ----------
        x_t : torch.Tensor
            Current samples.
        t : torch.Tensor
            Current time.

        Returns
        -------
        torch.Tensor
            Guidance term (zeros if no augmenter).
        """
        if self._augmenter is None:
            return torch.zeros_like(x_t)

        # Call augmenter
        h = self._augmenter(x_t, t)

        # Accumulate excess work if available
        if hasattr(self._augmenter, 'excess_work'):
            self._excess_work = self._excess_work + self._augmenter.excess_work

        return h


def sample_with_guidance(
    score_network: torch.nn.Module,
    beta_fn: Callable,
    augmenter: Optional[Callable] = None,
    n_samples: int = 1000,
    dt: float = 0.01,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    probability_flow: bool = False,
) -> torch.Tensor:
    """
    Convenience function to sample from a diffusion model with optional guidance.

    Parameters
    ----------
    score_network : torch.nn.Module
        Trained score network.
    beta_fn : callable
        Beta schedule.
    augmenter : callable, optional
        Guidance augmenter.
    n_samples : int
        Number of samples.
    dt : float
        Integration time step.
    device : str or torch.device
        Device for computations.
    seed : int, optional
        Random seed.
    probability_flow : bool
        Use ODE instead of SDE.

    Returns
    -------
    torch.Tensor
        Generated samples of shape (n_samples, 1, 1).
    """
    sampler = VPSDESampler(
        score_network=score_network,
        beta_fn=beta_fn,
        dt=dt,
        device=device,
        probability_flow=probability_flow,
    )
    sampler.augmenter = augmenter

    return sampler(n_samples=n_samples, seed=seed)

