"""NMR observables for protein benchmarks.

Provides the ``3J(HN-HA)`` scalar coupling observable computed from backbone
geometry via the Karplus equation, along with the backbone dihedral helpers it
relies on and an NMR Q-factor metric.
"""

from typing import Sequence, Union

import numpy as np
import torch


def compute_dihedral(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Signed dihedral (radians) between planes (p1, p2, p3) and (p2, p3, p4),
    using the b2 axis (p3 - p2) as the rotation hinge.
    """
    b2 = p3 - p2
    b2_norm = b2 / (b2.norm(dim=-1, keepdim=True) + eps)

    v0 = p1 - p2
    v1 = p4 - p3

    v0p = v0 - (v0 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm
    v1p = v1 - (v1 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm

    x = (v0p * v1p).sum(dim=-1)
    y = (torch.cross(b2_norm, v0p, dim=-1) * v1p).sum(dim=-1)

    return torch.atan2(y, x)


def backbone_phi(
    ca: torch.Tensor,
    Q: torch.Tensor,
    pN: torch.Tensor,
    pC: torch.Tensor,
) -> torch.Tensor:
    """Compute the backbone phi torsion from Cα positions and local frames."""
    N = ca + torch.einsum("...ij,j->...i", Q, pN)
    C = ca + torch.einsum("...ij,j->...i", Q, pC)
    C_prev = torch.roll(C, shifts=1, dims=1)
    phi = compute_dihedral(C_prev, N, ca, C)
    return phi


class ThreeJHNHA:
    """
    Compute 3J(HN-HA) via the backbone phi torsion

        phi_i = dihedral(C'_{i-1}, N_i, Ca_i, C'_i)

    and the Karplus relation

        J = A cos^2(phi - theta) + B cos(phi - theta) + C
    """

    def __init__(
        self,
        A: float = 7.09,
        B: float = -1.42,
        C: float = 1.55,
        theta: float = -60.0,  # degrees
        pN: Union[Sequence[float], torch.Tensor] = (-0.526, 1.363, 0.0),  # Ca->N (A)
        pC: Union[Sequence[float], torch.Tensor] = (1.526, 0.000, 0.0),  # Ca->C (A)
        device: Union[str, torch.device] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
    ):
        self.device = (
            torch.device("cuda:0")
            if device is None and torch.cuda.is_available()
            else torch.device(device or "cpu")
        )
        self.dtype = dtype

        self.A, self.B, self.C = A, B, C
        self.theta = theta

        self.pN = torch.as_tensor(pN, dtype=self.dtype, device=self.device)
        self.pC = torch.as_tensor(pC, dtype=self.dtype, device=self.device)

    def __call__(self, r: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        r : torch.Tensor
            Cα positions of shape (n_conf, n_res, 3). ASSUMES UNITS ARE IN NM.
        Q : torch.Tensor
            Local frames at each Cα, shape (n_conf, n_res, 3, 3).

        Returns
        -------
        torch.Tensor
            3J(HN-HA) couplings of shape (n_conf, n_res) (NaN where undefined).
        """
        n_conf, n_res, _ = r.shape

        r = r * 1e1  # Convert nm to Angstroms

        phi = backbone_phi(r, Q, self.pN, self.pC)

        theta_rad = torch.deg2rad(torch.tensor(self.theta))
        phi_shifted = phi + theta_rad

        cos_phi = torch.cos(phi_shifted)
        J = self.A * cos_phi**2 + self.B * cos_phi + self.C

        return J

    def expectation(self, observables_per_sample: torch.Tensor) -> torch.Tensor:
        return observables_per_sample.mean(dim=0, keepdims=True)


def calculate_quality_factor(
    d_exp: torch.Tensor,
    d_calc: torch.Tensor,
    weights: torch.Tensor = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute the NMR Q-factor between experimental and calculated observables.

        Q = sqrt( sum_i w_i * (d_calc_i - d_exp_i)^2  /  sum_i w_i * (d_exp_i)^2 )

    Parameters
    ----------
    d_exp : torch.Tensor
        Experimental data.
    d_calc : torch.Tensor
        Predicted data, same shape as ``d_exp``.
    weights : torch.Tensor, optional
        Non-negative weights broadcastable to ``d_exp``. If None, all weights
        are 1.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Scalar Q-factor rounded to 3 decimals.
    """
    d_exp = d_exp.reshape(-1)
    d_calc = d_calc.reshape(-1)

    if weights is None:
        w = torch.ones_like(d_exp)
    else:
        w = weights.reshape(-1)

    num = torch.sum(w * (d_calc - d_exp) ** 2)
    den = torch.sum(w * (d_exp) ** 2)

    return np.round(torch.sqrt(num / (den + eps)).item(), 3)
