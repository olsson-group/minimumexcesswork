"""Compatibility helpers for loading trained OGGM toy DDPM checkpoints."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import dill
import torch
from torch import nn


class OGGMScoreBasedDDPM(nn.Module):
    """
    Score network matching ``oggm.model.base.ScoreBasedDDPM``.

    The historical OGGM toy DDPM stores a scalar sinusoidal time embedding
    ``[cos(k t), sin(k t)]`` regardless of ``time_embedding_dim``.  This class
    preserves that checkpoint architecture so pretrained OGGM weights can be
    used as the MEW-OG base model.
    """

    def __init__(
        self,
        n_atoms: int = 1,
        dim: int = 1,
        time_embedding_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 1,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.dim = dim
        self.time_embedding_dim = time_embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.config = {
            "n_atoms": n_atoms,
            "dim": dim,
            "time_embedding_dim": time_embedding_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
        }

        input_dim = n_atoms * dim + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        t_embedded = self.time_embedding(x_flat, t)
        output = self.network(torch.cat([x_flat, t_embedded], dim=1))
        return output.view(x.shape)

    def time_embedding(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t = t.squeeze().view(-1)
        t_embedded = self.time_embedding_dim * t[..., None]
        t_embedded = torch.cat((t_embedded.cos(), t_embedded.sin()), dim=-1)
        if t_embedded.shape[0] == 1 and x.shape[0] > 1:
            t_embedded = t_embedded.expand(x.shape[0], -1)
        return t_embedded

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OGGMScoreBasedDDPM":
        return cls(
            n_atoms=config.get("n_atoms", 1),
            dim=config.get("dim", 1),
            time_embedding_dim=config.get("time_embedding_dim", 3),
            hidden_dim=config.get("hidden_dim", 64),
            out_dim=config.get("out_dim", config.get("dim", 1)),
        )


def load_oggm_score_model(
    checkpoint_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "cpu",
) -> OGGMScoreBasedDDPM:
    """Load an OGGM toy DDPM checkpoint into the compatibility model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _torch_load_oggm_checkpoint(checkpoint_path, device=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'model_state_dict'")

    model_config = _infer_config_from_state_dict(state_dict)
    if config:
        model_config.update(config)

    model = OGGMScoreBasedDDPM.from_config(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _torch_load_oggm_checkpoint(
    checkpoint_path: Path,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    pickle_module = SimpleNamespace(
        __name__="mew_og_oggm_checkpoint_pickle",
        Unpickler=_StateDictOnlyUnpickler,
    )
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            pickle_module=pickle_module,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device, pickle_module=pickle_module)


class _IgnoredPickleObject:
    """Placeholder for pickled model/optimizer objects not needed by MEW-OG."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StateDictOnlyUnpickler(dill.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, ModuleNotFoundError, AttributeError):
            return _IgnoredPickleObject


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    first_weight = state_dict["network.0.weight"]
    last_weight = state_dict["network.6.weight"]
    input_dim = first_weight.shape[1]
    hidden_dim = first_weight.shape[0]
    out_dim = last_weight.shape[0]
    n_features = input_dim - 2
    return {
        "n_atoms": 1,
        "dim": n_features,
        "time_embedding_dim": 3,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
    }
