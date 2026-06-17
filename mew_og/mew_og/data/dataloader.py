"""Simple data loading utilities (no torch_geometric dependency)."""

from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """
    A simple dataset for trajectory data.

    Each item returns (x, index) where x is the data point and index is its position.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        The trajectory data of shape (n_samples,) or (n_samples, n_features).
    weights : torch.Tensor or np.ndarray, optional
        Optional weights for each sample.
    """

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Ensure at least 2D
        if data.dim() == 1:
            data = data.unsqueeze(1)

        self.data = data

        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights).float()
            self.weights = weights
        else:
            self.weights = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single data point.

        Returns
        -------
        dict
            Dictionary with keys 'x' (data), 'index' (position), and optionally 'weight'.
        """
        item = {
            "x": self.data[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if self.weights is not None:
            item["weight"] = self.weights[idx]
        return item


def collate_fn(batch: list) -> dict:
    """
    Collate function for batching TrajectoryDataset items.

    Parameters
    ----------
    batch : list
        List of items from TrajectoryDataset.

    Returns
    -------
    dict
        Batched data with stacked tensors.
    """
    x = torch.stack([item["x"] for item in batch], dim=0)
    index = torch.stack([item["index"] for item in batch], dim=0)
    result = {"x": x, "index": index}

    if "weight" in batch[0]:
        result["weight"] = torch.stack([item["weight"] for item in batch], dim=0)

    return result


def create_data_loader(
    data: Union[torch.Tensor, np.ndarray],
    batch_size: int = 256,
    shuffle: bool = True,
    weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for trajectory data.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        The trajectory data.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the data.
    weights : torch.Tensor or np.ndarray, optional
        Optional weights for each sample.
    **kwargs
        Additional arguments passed to DataLoader.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader.
    """
    dataset = TrajectoryDataset(data, weights=weights)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **kwargs,
    )

