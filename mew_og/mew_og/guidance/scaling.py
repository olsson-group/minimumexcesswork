"""Time-dependent scaling functions for guidance."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class ScalingFunction(ABC, nn.Module):
    """
    Abstract base class for time-dependent scaling functions.

    These functions control how strongly the guidance term affects the
    sampling process at different times t in [0, 1].
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scaling factor at time t.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1].

        Returns
        -------
        torch.Tensor
            Scaling factor(s).
        """
        pass

    @abstractmethod
    def definite_integral(self) -> torch.Tensor:
        """
        Compute the definite integral of the scaling function from 0 to 1.

        This is used for regularization (excess work penalty).

        Returns
        -------
        torch.Tensor
            Integral value.
        """
        pass


class ExponentialScaling(ScalingFunction):
    """
    Exponential scaling function: alpha(t) = a * exp(-b * t).

    The sampler integrates from t=1 (noise) to t=0 (data), so this scaling is
    strongest near the data distribution and weakest near the noise prior.

    Parameters
    ----------
    a : float
        Amplitude parameter.
    b : float
        Decay rate parameter.
    """

    def __init__(self, a: float = 1.0, b: float = 5.0):
        super().__init__()
        self._a = nn.Parameter(torch.tensor([a]))
        self._b = nn.Parameter(torch.tensor([b]))

    @property
    def a(self) -> torch.Tensor:
        return self._a

    @property
    def b(self) -> torch.Tensor:
        return self._b

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha(t) = a * exp(-b * t).

        Parameters
        ----------
        t : torch.Tensor
            Time values.

        Returns
        -------
        torch.Tensor
            Scaling factors.
        """
        return self._a * torch.exp(-self._b * t)

    def definite_integral(self) -> torch.Tensor:
        """
        Compute integral of a * exp(-b * t) from 0 to 1.

        integral = (a / b) * (1 - exp(-b))

        Returns
        -------
        torch.Tensor
            Integral value.
        """
        return (self._a / self._b) * (1 - torch.exp(-self._b))


class LinearScaling(ScalingFunction):
    """
    Linear scaling function: alpha(t) = a * (1 - t) + b.

    Parameters
    ----------
    a : float
        Slope parameter.
    b : float
        Intercept parameter.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        super().__init__()
        self._a = nn.Parameter(torch.tensor([a]))
        self._b = nn.Parameter(torch.tensor([b]))

    @property
    def a(self) -> torch.Tensor:
        return self._a

    @property
    def b(self) -> torch.Tensor:
        return self._b

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha(t) = a * (1 - t) + b.

        Parameters
        ----------
        t : torch.Tensor
            Time values.

        Returns
        -------
        torch.Tensor
            Scaling factors.
        """
        return self._a * (1 - t) + self._b

    def definite_integral(self) -> torch.Tensor:
        """
        Compute integral of a * (1 - t) + b from 0 to 1.

        integral = a/2 + b

        Returns
        -------
        torch.Tensor
            Integral value.
        """
        return self._a / 2 + self._b


class ConstantScaling(ScalingFunction):
    """
    Constant scaling: alpha(t) = c.

    Parameters
    ----------
    c : float
        Constant scaling value.
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self._c = nn.Parameter(torch.tensor([c]))

    @property
    def c(self) -> torch.Tensor:
        return self._c

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return constant scaling.

        Parameters
        ----------
        t : torch.Tensor
            Time values (ignored).

        Returns
        -------
        torch.Tensor
            Constant scaling value.
        """
        return self._c.expand_as(t) if t.dim() > 0 else self._c

    def definite_integral(self) -> torch.Tensor:
        """
        Integral of constant c from 0 to 1 is c.

        Returns
        -------
        torch.Tensor
            Integral value (equals c).
        """
        return self._c

