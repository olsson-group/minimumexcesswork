"""Path utilities."""

from pathlib import Path
from typing import Optional, Union


def get_project_root(marker_file: str = "pyproject.toml") -> Path:
    """
    Find the project root by searching upward for a marker file.

    Parameters
    ----------
    marker_file : str
        Name of the file that marks the project root.

    Returns
    -------
    Path
        Path to the project root directory.

    Raises
    ------
    FileNotFoundError
        If the marker file is not found in any parent directory.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker_file).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (looking for {marker_file})"
    )


def resolve_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve a path, optionally relative to a base directory.

    Parameters
    ----------
    path : str or Path
        The path to resolve.
    base_dir : str or Path, optional
        Base directory for relative paths. If None, uses current working directory.

    Returns
    -------
    Path
        The resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (Path(base_dir) / path).resolve()
    return path.resolve()

