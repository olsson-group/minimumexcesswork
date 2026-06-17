"""Tensor conversion and manipulation utilities."""

from typing import Union

import numpy as np
import torch


def to_tensor(
    data: Union[np.ndarray, torch.Tensor, list],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Convert data to a torch tensor.

    Parameters
    ----------
    data : array-like
        Input data (numpy array, torch tensor, or list).
    dtype : torch.dtype
        Target data type.
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


def to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert data to a numpy array.

    Parameters
    ----------
    data : array-like
        Input data (numpy array or torch tensor).

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def filter_outliers(
    tensor: torch.Tensor,
    threshold: float = 10.0,
    num_std_dev: int = 3,
    extreme_value: float = 0.0,
) -> torch.Tensor:
    """
    Filter extreme values in a tensor by setting them to a specified value.

    Values are considered extreme if they exceed the threshold in absolute value
    or lie outside `num_std_dev` standard deviations from the mean.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    threshold : float
        Absolute value threshold for initial filtering.
    num_std_dev : int
        Number of standard deviations for statistical filtering.
    extreme_value : float
        Value to replace extreme values with.

    Returns
    -------
    torch.Tensor
        Tensor with extreme values replaced.
    """
    # Initial mask: finite and within threshold
    feasible_mask = (tensor.abs() < threshold) & torch.isfinite(tensor)
    feasible_tensor = tensor[feasible_mask]

    if feasible_tensor.numel() > 0:
        mean = feasible_tensor.mean()
        std = feasible_tensor.std()
        lower = mean - num_std_dev * std
        upper = mean + num_std_dev * std
        final_mask = (tensor >= lower) & (tensor <= upper) & feasible_mask
    else:
        final_mask = feasible_mask

    result = tensor.clone()
    result[~final_mask] = extreme_value
    return result

