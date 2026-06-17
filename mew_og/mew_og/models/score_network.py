"""Score-based neural network for DDPM."""

from typing import Dict, Any

import torch
from torch import nn


class ScoreBasedDDPM(nn.Module):
    """
    A simple score-based neural network for denoising diffusion models.

    This network takes noisy samples x_t and time t, and predicts the score
    (gradient of log probability) at that point.

    Architecture: MLP with sinusoidal time embedding.

    Parameters
    ----------
    n_atoms : int
        Number of atoms (for molecular data, use 1 for 1D toy).
    dim : int
        Dimension per atom.
    time_embedding_dim : int
        Dimension of time embedding.
    hidden_dim : int
        Hidden layer dimension.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        n_atoms: int = 1,
        dim: int = 1,
        time_embedding_dim: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.dim = dim
        self.time_embedding_dim = time_embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Store config for checkpointing
        self.config = {
            "n_atoms": n_atoms,
            "dim": dim,
            "time_embedding_dim": time_embedding_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
        }

        # Input: x (flattened) + time embedding (2 * time_embedding_dim for sin/cos)
        input_dim = n_atoms * dim + 2 * time_embedding_dim
        output_dim = n_atoms * dim

        # Build MLP
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the score network.

        Parameters
        ----------
        x : torch.Tensor
            Input samples of shape (batch_size, n_atoms, dim).
        t : torch.Tensor
            Time values of shape (batch_size,) or broadcastable.

        Returns
        -------
        torch.Tensor
            Score predictions of shape (batch_size, n_atoms, dim).
        """
        batch_size = x.shape[0]
        original_shape = x.shape

        # Flatten spatial dimensions
        x_flat = x.view(batch_size, -1)

        # Create time embedding
        t_embed = self._time_embedding(x_flat, t)

        # Concatenate and forward
        x_and_t = torch.cat([x_flat, t_embed], dim=-1)
        output = self.network(x_and_t)

        # Reshape to original shape
        return output.view(original_shape)

    def _time_embedding(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (used for shape/device).
        t : torch.Tensor
            Time values.

        Returns
        -------
        torch.Tensor
            Time embedding of shape (batch_size, 2 * time_embedding_dim).
        """
        # Ensure t has correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1)

        # Frequency encoding
        freq = self.time_embedding_dim * t.unsqueeze(-1)

        # Sinusoidal encoding
        t_embed = torch.cat([freq.cos(), freq.sin()], dim=-1)

        # Expand to batch size if needed
        if t_embed.shape[0] == 1 and x.shape[0] > 1:
            t_embed = t_embed.expand(x.shape[0], -1)

        return t_embed

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ScoreBasedDDPM":
        """
        Create a model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Model configuration.

        Returns
        -------
        ScoreBasedDDPM
            Model instance.
        """
        return cls(
            n_atoms=config.get("n_atoms", 1),
            dim=config.get("dim", 1),
            time_embedding_dim=config.get("time_embedding_dim", 3),
            hidden_dim=config.get("hidden_dim", 64),
            n_layers=config.get("n_layers", 3),
        )

