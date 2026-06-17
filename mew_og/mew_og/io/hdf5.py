"""Simple HDF5 I/O utilities (no external dependencies beyond h5py)."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import torch


def read_hdf5(
    file_path: Union[str, Path],
    dataset_name: str,
    group_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Read a dataset from an HDF5 file and return as a torch.Tensor.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    dataset_name : str
        Name of the dataset to read.
    group_name : str, optional
        Name of the group containing the dataset. If None, reads from root.

    Returns
    -------
    torch.Tensor
        The data as a torch tensor.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        if group_name is not None:
            group = f[group_name]
        else:
            group = f
        data = np.array(group[dataset_name])
    return torch.from_numpy(data)


def write_hdf5(
    file_path: Union[str, Path],
    data_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    group_name: Optional[str] = None,
    mode: str = "a",
) -> None:
    """
    Write data to an HDF5 file.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    data_dict : dict
        Dictionary mapping dataset names to data (numpy arrays or torch tensors).
    group_name : str, optional
        Name of the group to write to. If None, writes to root.
    mode : str
        File mode ('w' for overwrite, 'a' for append/update).
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, mode) as f:
        if group_name is not None:
            if group_name not in f:
                group = f.create_group(group_name)
            else:
                group = f[group_name]
        else:
            group = f

        for name, data in data_dict.items():
            # Convert torch tensors to numpy
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

            # Delete existing dataset if present
            if name in group:
                del group[name]

            group.create_dataset(name, data=data)


class HDFLoader:
    """
    Simple HDF5 file loader that caches data in memory.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._data: Dict[str, torch.Tensor] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all datasets from the HDF5 file."""
        with h5py.File(self.file_path, "r") as f:
            self._recursive_load(f, "")

    def _recursive_load(self, group: h5py.Group, path: str) -> None:
        """Recursively load all datasets from a group."""
        for key in group.keys():
            item = group[key]
            item_path = f"{path}/{key}".lstrip("/")
            if isinstance(item, h5py.Dataset):
                self._data[item_path] = torch.from_numpy(np.array(item))
            elif isinstance(item, h5py.Group):
                self._recursive_load(item, item_path)

    def read(self, dataset_name: str) -> torch.Tensor:
        """
        Read a dataset by name.

        Parameters
        ----------
        dataset_name : str
            Name/path of the dataset.

        Returns
        -------
        torch.Tensor
            The data as a torch tensor.
        """
        if dataset_name in self._data:
            return self._data[dataset_name]
        raise KeyError(f"Dataset '{dataset_name}' not found in {self.file_path}")

    def keys(self) -> list:
        """Return all dataset keys."""
        return list(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.read(key)

