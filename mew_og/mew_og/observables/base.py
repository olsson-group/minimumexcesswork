"""Base classes for observable functions."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class ObservableBase(ABC):
    """
    Abstract base class for observable functions.

    An observable is a function that maps samples (configurations) to
    observable values. These are used both for computing experimental
    constraints and for guiding the generative model.

    Attributes
    ----------
    params : torch.Tensor
        Parameters of the observable function.
    """

    def __init__(self, params: Union[torch.Tensor, np.ndarray, None] = None):
        """
        Initialize the observable.

        Parameters
        ----------
        params : torch.Tensor or np.ndarray, optional
            Parameters for the observable function.
        """
        if params is not None:
            if isinstance(params, np.ndarray):
                params = torch.from_numpy(params).float()
            self.params = params
        else:
            self.params = torch.empty(0)

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the observable on samples.

        Parameters
        ----------
        x : torch.Tensor
            Input samples of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Observable values of shape (n_samples, n_observables).
        """
        pass

    def expectation(self, observables_per_sample: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation (mean) of the observable.

        Parameters
        ----------
        observables_per_sample : torch.Tensor
            Observable values for each sample.

        Returns
        -------
        torch.Tensor
            Mean observable value.
        """
        return observables_per_sample.mean(dim=0, keepdim=True)


class PolynomialObservable(ObservableBase):
    """
    Polynomial observable function.

    Computes f(x) = sum_i a_i * (x + b_i)^deg

    Parameters
    ----------
    params : torch.Tensor or np.ndarray
        Parameters of shape (n_terms, 2) where each row is [a_i, b_i].
    deg : int
        Degree of the polynomial.
    """

    def __init__(
        self,
        params: Union[torch.Tensor, np.ndarray, None] = None,
        deg: int = 2,
    ):
        super().__init__(params)
        self.deg = deg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the polynomial observable.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Observable values of shape (n_samples, 1).
        """
        x = x.view(-1)
        result = torch.zeros_like(x, dtype=torch.float32)

        for a, b in self.params:
            result = result + a * (x + b) ** self.deg

        return result.unsqueeze(-1)

    def __repr__(self) -> str:
        terms = [f"{a.item():.2f}*(x + {b.item():.2f})^{self.deg}" for a, b in self.params]
        return f"PolynomialObservable: {' + '.join(terms)}"


class IdentityObservable(ObservableBase):
    """
    Identity observable: f(x) = x.
    """

    def __init__(self):
        super().__init__(None)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return input unchanged (reshaped to (n_samples, 1)).

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Same values reshaped to (n_samples, 1).
        """
        return x.view(-1, 1)

