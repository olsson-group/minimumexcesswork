"""Checkpoint saving and loading utilities."""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch


def save_checkpoint(
    file_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    extra_data: Optional[dict] = None,
) -> None:
    """
    Save a model checkpoint.

    Parameters
    ----------
    file_path : str or Path
        Path to save the checkpoint.
    model : torch.nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer, optional
        The optimizer to save.
    epoch : int
        Current epoch number.
    extra_data : dict, optional
        Additional data to save in the checkpoint.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": getattr(model, "config", {}),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if extra_data is not None:
        state["extra"] = extra_data

    torch.save(state, file_path)


def load_checkpoint(
    file_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Optional[torch.nn.Module], Optional[torch.optim.Optimizer], int, dict]:
    """
    Load a model checkpoint.

    Parameters
    ----------
    file_path : str or Path
        Path to the checkpoint file.
    model : torch.nn.Module, optional
        Model instance to load weights into. If None, only returns state dict.
    optimizer : torch.optim.Optimizer, optional
        Optimizer instance to load state into.
    device : str or torch.device
        Device to map the checkpoint to.

    Returns
    -------
    model : torch.nn.Module or None
        The model with loaded weights (or None if no model provided).
    optimizer : torch.optim.Optimizer or None
        The optimizer with loaded state (or None if no optimizer provided).
    epoch : int
        The epoch at which the checkpoint was saved.
    extra : dict
        Any extra data stored in the checkpoint.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")

    checkpoint = torch.load(file_path, map_location=device)

    epoch = checkpoint.get("epoch", 0)
    extra = checkpoint.get("extra", {})

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, epoch, extra

