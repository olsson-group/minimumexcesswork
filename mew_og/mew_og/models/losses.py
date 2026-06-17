"""Loss functions for DDPM training."""

from typing import Callable

import torch


def ddpm_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    beta_integral_fn: Callable[[torch.Tensor], torch.Tensor],
    t_min: float = 1e-4,
    t_max: float = 1.0,
) -> torch.Tensor:
    """
    Compute the VP-DDPM denoising score matching loss.

    This loss trains the score network to predict the gradient of log p(x_t | x_0).

    Parameters
    ----------
    model : torch.nn.Module
        Score network that predicts score(x_t, t).
    x : torch.Tensor
        Clean samples x_0 of shape (batch_size, ...).
    beta_integral_fn : callable
        Function that computes the integral of beta from 0 to t.
    t_min : float
        Minimum time value.
    t_max : float
        Maximum time value.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    return ddpm_loss_per_sample(model, x, beta_integral_fn, t_min, t_max).mean()


def ddpm_loss_per_sample(
    model: torch.nn.Module,
    x: torch.Tensor,
    beta_integral_fn: Callable[[torch.Tensor], torch.Tensor],
    t_min: float = 1e-4,
    t_max: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-sample VP-DDPM loss.

    Parameters
    ----------
    model : torch.nn.Module
        Score network.
    x : torch.Tensor
        Clean samples x_0 of shape (batch_size, ...).
    beta_integral_fn : callable
        Function that computes the integral of beta from 0 to t.
    t_min : float
        Minimum time value.
    t_max : float
        Maximum time value.

    Returns
    -------
    torch.Tensor
        Per-sample loss values of shape (batch_size,).
    """
    batch_size = x.shape[0]

    # Sample random times uniformly in [t_min, t_max]
    t = torch.rand(batch_size, device=x.device, dtype=x.dtype) * (t_max - t_min) + t_min

    # Reshape t for broadcasting with x
    t_expanded = t.view(batch_size, *([1] * (x.dim() - 1)))

    # Compute forward diffusion parameters
    beta_int = beta_integral_fn(t_expanded)
    mu_t = x * torch.exp(-0.5 * beta_int)
    sigma_sq_t = -torch.expm1(-beta_int)  # 1 - exp(-beta_int)

    # Sample noisy x_t
    noise = torch.randn_like(x)
    x_t = mu_t + torch.sqrt(sigma_sq_t) * noise

    # Compute true score: grad_x log p(x_t | x_0) = -(x_t - mu_t) / sigma_sq_t
    true_score = -(x_t - mu_t) / sigma_sq_t

    # Predict score
    predicted_score = model(x_t, t)

    # MSE loss per sample
    loss = ((predicted_score - true_score) ** 2).view(batch_size, -1).mean(dim=1)

    return loss


def simple_ddpm_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    beta_fn: Callable[[torch.Tensor], torch.Tensor],
    t_min: float = 1e-4,
    t_max: float = 1.0,
) -> torch.Tensor:
    """
    Simple epsilon-prediction DDPM loss.

    Trains the model to predict the noise added during forward diffusion.

    Parameters
    ----------
    model : torch.nn.Module
        Model that predicts noise epsilon.
    x : torch.Tensor
        Clean samples x_0.
    beta_fn : callable
        Beta schedule function.
    t_min : float
        Minimum time value.
    t_max : float
        Maximum time value.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    batch_size = x.shape[0]

    # Sample random times
    t = torch.rand(batch_size, device=x.device, dtype=x.dtype) * (t_max - t_min) + t_min
    t_expanded = t.view(batch_size, *([1] * (x.dim() - 1)))

    # Forward diffusion (simplified)
    beta_t = beta_fn(t_expanded)
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)  # Simplified; in practice use integral

    # For simplicity, use the continuous formula
    noise = torch.randn_like(x)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

    x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

    # Predict noise
    epsilon_pred = model(x_t, t)

    # MSE loss
    return torch.nn.functional.mse_loss(epsilon_pred, noise)

