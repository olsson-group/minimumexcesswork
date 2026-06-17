"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Union


def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load a JSON configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the JSON config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: dict, config_path: Union[str, Path]) -> None:
    """
    Save a configuration dictionary to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    config_path : str or Path
        Path to save the JSON file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

