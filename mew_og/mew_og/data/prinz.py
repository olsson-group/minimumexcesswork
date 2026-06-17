"""Prinz potential utilities for toy experiments."""

from typing import Union

import numpy as np
import torch


def prinz_potential(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Compute the Prinz potential.

    The Prinz potential is a 1D toy potential with 4 metastable states,
    commonly used for testing enhanced sampling and diffusion methods.

    V(x) = 4 * (x^8 + 0.8 * exp(-80*x^2) + 0.2 * exp(-80*(x-0.5)^2)
                + 0.5 * exp(-40*(x+0.5)^2))

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input coordinates.

    Returns
    -------
    torch.Tensor
        Potential energy values.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    return 4 * (
        x**8
        + 0.8 * torch.exp(-80 * x**2)
        + 0.2 * torch.exp(-80 * (x - 0.5) ** 2)
        + 0.5 * torch.exp(-40 * (x + 0.5) ** 2)
    )


def generate_prinz_trajectory(
    n_samples: int = 100000,
    dt: float = 1e-4,
    kT: float = 1.0,
    x0: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a trajectory on the Prinz potential using Langevin dynamics.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    dt : float
        Time step for integration.
    kT : float
        Temperature (in units where k_B = 1).
    x0 : float
        Initial position.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Trajectory of shape (n_samples,).
    """
    np.random.seed(seed)

    # Langevin dynamics: dx = -dV/dx * dt + sqrt(2*kT*dt) * noise
    gamma = 1.0  # friction coefficient
    noise_scale = np.sqrt(2 * kT * dt / gamma)

    trajectory = np.zeros(n_samples)
    x = x0

    for i in range(n_samples):
        # Compute force (negative gradient of potential)
        x_tensor = torch.tensor([x], dtype=torch.float32, requires_grad=True)
        V = prinz_potential(x_tensor)
        V.backward()
        force = -x_tensor.grad.item()

        # Euler-Maruyama step
        x = x + force * dt / gamma + noise_scale * np.random.randn()
        trajectory[i] = x

    return torch.from_numpy(trajectory).float()


def generate_deeptime_prinz_trajectory(
    n_samples: int = 100000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate Prinz samples using deeptime's default Prinz sampler settings.

    The only controlled value is the number of samples. The sampler itself is
    constructed as ``deeptime.data.prinz_potential()``.
    """
    try:
        from deeptime.data import prinz_potential as deeptime_prinz_potential
    except ImportError as exc:
        raise ImportError(
            "deeptime is required for the Prinz sampler. Install deeptime."
        ) from exc

    np.random.seed(seed)
    system = deeptime_prinz_potential()
    trajectory = system.trajectory([[0.0]], n_samples).squeeze()
    return torch.from_numpy(np.asarray(trajectory, dtype=np.float32))


def bias_trajectory(
    trajectory: Union[torch.Tensor, np.ndarray],
    coefficient: float = -4.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create a biased version of a trajectory by resampling with tilted weights.

    This simulates having data from a biased simulation/prior distribution.

    Parameters
    ----------
    trajectory : torch.Tensor or np.ndarray
        Original trajectory.
    coefficient : float
        Bias coefficient. Negative values bias toward lower x values.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Biased trajectory.
    """
    np.random.seed(seed)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()
    trajectory = np.squeeze(trajectory)

    # Compute tilting weights
    weights = trajectory - trajectory.min()
    linear_fn = coefficient * np.linspace(0, trajectory.max(), len(trajectory))
    weights = weights * linear_fn
    weights = np.exp(weights)
    weights = weights / weights.sum()

    # Resample according to weights
    resampled_indices = np.random.choice(
        len(trajectory), size=len(trajectory), p=weights
    )
    return torch.from_numpy(trajectory[resampled_indices]).float()

