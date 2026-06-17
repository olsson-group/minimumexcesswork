"""Seed utilities for reproducibility."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all relevant libraries.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_seed(max_value: int = 10000) -> int:
    """
    Generate a random seed value.

    Parameters
    ----------
    max_value : int
        Maximum value for the seed.

    Returns
    -------
    int
        A random seed value.
    """
    return random.randint(0, max_value)

