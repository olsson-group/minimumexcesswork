"""Gaussian Mixture Model observable for guidance."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from mew_og.observables.base import ObservableBase


class GaussianMixtureObservable(ObservableBase):
    """
    Gaussian Mixture Model observable.

    This observable evaluates the weighted sum of Gaussian PDFs plus a linear term,
    which can be used to target specific regions of the configuration space.

    The function is:
        f(x) = sum_i w_i * N(x; mu_i, sigma_i^2) + x

    Parameters
    ----------
    params : torch.Tensor or np.ndarray
        Parameters of shape (n_components, 3) where each row is [mean, variance, weight].
    """

    def __init__(self, params: Union[torch.Tensor, np.ndarray, None] = None):
        super().__init__(params)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the GMM observable.

        Parameters
        ----------
        x : torch.Tensor
            Input samples of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        torch.Tensor
            Observable values of shape (n_samples, 1).
        """
        x = x.view(-1).float()
        result = torch.zeros_like(x, dtype=torch.float32)

        for mean, variance, weight in self.params:
            # Gaussian PDF
            pdf = (1 / torch.sqrt(2 * torch.pi * variance)) * torch.exp(
                -0.5 * ((x - mean) ** 2) / variance
            )
            result = result + weight * pdf

        # Add linear term
        result = result + x

        return result.unsqueeze(-1)

    def __repr__(self) -> str:
        n_components = len(self.params) if self.params is not None else 0
        return f"GaussianMixtureObservable(n_components={n_components})"


def fit_gmm(
    data: Union[torch.Tensor, np.ndarray],
    n_components: int = 4,
    random_state: int = 42,
    use_histogram_weighting: bool = True,
    n_bins: int = 50,
) -> GaussianMixture:
    """
    Fit a Gaussian Mixture Model to data.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Input data of shape (n_samples,).
    n_components : int
        Number of Gaussian components.
    random_state : int
        Random seed for reproducibility.
    use_histogram_weighting : bool
        If True, weight samples by histogram density for better mode capture.
    n_bins : int
        Number of bins for histogram weighting.

    Returns
    -------
    GaussianMixture
        Fitted sklearn GaussianMixture model.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data = np.squeeze(data)

    if use_histogram_weighting:
        # Create histogram-weighted data for better mode capture
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        data_weighted = np.repeat(bin_centers, (hist * len(data)).astype(int))
        fit_data = data_weighted.reshape(-1, 1)
    else:
        fit_data = data.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(fit_data)

    return gmm


def gmm_to_params(gmm: GaussianMixture) -> np.ndarray:
    """
    Extract parameters from a fitted GaussianMixture.

    Parameters
    ----------
    gmm : GaussianMixture
        Fitted sklearn GaussianMixture model.

    Returns
    -------
    np.ndarray
        Parameters of shape (n_components, 3) with columns [mean, variance, weight].
    """
    return np.column_stack((
        gmm.means_.flatten(),
        gmm.covariances_.flatten(),
        gmm.weights_,
    ))


def save_gmm_params(
    gmm: Union[GaussianMixture, np.ndarray],
    file_path: Union[str, Path],
) -> None:
    """
    Save GMM parameters to a .npy file.

    Parameters
    ----------
    gmm : GaussianMixture or np.ndarray
        Either a fitted GaussianMixture or parameters array.
    file_path : str or Path
        Output file path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(gmm, GaussianMixture):
        params = gmm_to_params(gmm)
    else:
        params = gmm

    np.save(file_path, params)


def load_gmm_params(file_path: Union[str, Path]) -> GaussianMixtureObservable:
    """
    Load GMM parameters and create an observable.

    Parameters
    ----------
    file_path : str or Path
        Path to the .npy file containing GMM parameters.

    Returns
    -------
    GaussianMixtureObservable
        Observable initialized with the loaded parameters.
    """
    file_path = Path(file_path)
    params = np.load(file_path)
    return GaussianMixtureObservable(params=params)

