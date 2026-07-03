"""Observable-guidance augmenter for BioEmu protein samples.

:class:`BioEmuAugmenter` extends the base :class:`mew_og.guidance.augmenter.Augmenter`
to operate on protein conformations represented as a tuple ``(r, Q)`` where

* ``r`` are Cα positions of shape ``(n_conf, n_res, 3)``
* ``Q`` are per-residue rotation matrices of shape ``(n_conf, n_res, 3, 3)``

The guidance force is computed on the SO(3) manifold and mapped back to the
tangent space, and an accumulated excess-work term is tracked for the
minimum-excess-work regularization.
"""

import re
from typing import Callable, List, Tuple

import torch

from mew_og.guidance.augmenter import Augmenter


def calculate_excess_work(
    guiding_strength: torch.Tensor, lambda_observables: torch.Tensor
) -> torch.Tensor:
    """Excess work: ``sum(strength^2 * mean(components^2, dim=0))``."""
    return torch.sum(
        guiding_strength**2 * torch.mean(lambda_observables**2, dim=0)
    )


class BioEmuAugmenter(Augmenter):
    """Guidance augmenter for ``(r, Q)`` protein samples.

    Parameters
    ----------
    experimental_data : list of StaticExperiment
        Experiments carrying the shared observable function and per-residue
        targets. ``exp.name`` (or ``exp.resid``) is used to select observable
        columns.
    scaling_function : list of ScalingFunction
        One time-dependent scaling function per observable.
    lambdas : torch.Tensor
        Reweighting Lagrange multipliers.
    normalization : optional
        Accepted for interface compatibility; unused (kept ``None``).
    device : str or torch.device
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.
    """

    def __init__(
        self,
        experimental_data: List = None,
        scaling_function: List[Callable] = None,
        lambdas: torch.Tensor = None,
        normalization=None,
        use_global_scaling: bool = False,
        dtype: torch.dtype = torch.float32,
        device="cpu",
    ):
        super().__init__(
            experimental_data=experimental_data,
            lambdas=lambdas,
            scaling_function=scaling_function,
            device=device,
            dtype=dtype,
        )
        self.normalization = normalization
        self.use_global_scaling = use_global_scaling
        self._excess_work_accum = torch.zeros(1, device=device)
        self.obs_idx = self._build_obs_idx(device=device)

    @property
    def experimental_data(self) -> List:
        """Experiments backing this augmenter."""
        return self._experimental_data

    # -- excess-work accumulator (mutable, unlike the base read-only property) --
    @property
    def excess_work(self) -> torch.Tensor:
        return self._excess_work_accum

    @excess_work.setter
    def excess_work(self, value: torch.Tensor) -> None:
        self._excess_work_accum = value

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the MEW guidance force for a batch of ``(r, Q)`` samples.

        Parameters
        ----------
        x : tuple of torch.Tensor
            ``(r, Q)`` with ``r`` shape ``(B, n_res, 3)`` and ``Q`` shape
            ``(B, n_res, 3, 3)``.
        t : torch.Tensor
            Diffusion time, constant across the batch.

        Returns
        -------
        tuple of torch.Tensor
            ``(force_r, force_Q)`` matching the shape of ``r``.
        """
        r, Q = x
        batch_size = t.shape[0]
        r_flat = r.reshape(batch_size, -1, 3)
        Q_flat = Q.reshape(batch_size, -1, 3, 3)
        t_scalar = t[0] if t.ndim > 0 else t
        assert torch.unique(t) == t[0], "t must be constant across the batch"

        full_obs = self.observables_function(r_flat, Q_flat)  # (B, N_obs_total)
        observables_per_sample = full_obs.index_select(
            dim=1, index=self.obs_idx
        )  # (B, N_obs)

        eta_t_list = [f(t_scalar) for f in self.scaling_function]
        if len(eta_t_list) != observables_per_sample.shape[1]:
            raise ValueError(
                f"Number of scaling functions ({len(eta_t_list)}) must match "
                f"number of observables ({observables_per_sample.shape[1]})"
            )

        processed_eta_t_list = []
        for eta_val in eta_t_list:
            eta_tensor = torch.as_tensor(eta_val, device=r.device, dtype=self.dtype)
            if eta_tensor.ndim > 0:
                eta_tensor = eta_tensor.squeeze()
            processed_eta_t_list.append(eta_tensor)

        eta_t = torch.stack(processed_eta_t_list).unsqueeze(0).expand(
            observables_per_sample.shape[0], -1
        )  # (B, N_obs)

        obs_scaled = observables_per_sample * eta_t
        lambdas = torch.as_tensor(self.lambdas, device=obs_scaled.device, dtype=self.dtype)
        weighted = (obs_scaled * lambdas).sum(dim=1)

        grad_r, grad_Q = torch.autograd.grad(
            outputs=weighted,
            inputs=(r, Q),
            grad_outputs=torch.ones_like(weighted),
            create_graph=False,
            retain_graph=False,
        )

        G = grad_Q
        QtG = torch.matmul(Q.transpose(-2, -1), G)
        skew = 0.5 * (QtG - QtG.transpose(-2, -1))
        grad_Q_tangent = self.vee_map(skew)

        force_r = -grad_r
        force_Q = -grad_Q_tangent

        self.accumulate_excess_work(force_r, force_Q, eta_t)

        force_r = force_r.reshape(r.shape)
        force_Q = force_Q.reshape(r.shape)

        return force_r, force_Q

    def accumulate_excess_work(self, force_r, force_Q, eta_t) -> None:
        """Accumulate the excess-work regularization term for this step."""
        all_components = torch.cat(
            [
                force_r.reshape(force_r.shape[0], -1),
                force_Q.reshape(force_Q.shape[0], -1),
            ],
            dim=1,
        )

        if isinstance(eta_t, torch.Tensor) and eta_t.ndim > 0:
            alpha_eff = eta_t.mean()
        else:
            alpha_eff = eta_t

        all_components = all_components / alpha_eff
        ew = calculate_excess_work(alpha_eff, all_components).detach().item()
        self.excess_work = self.excess_work + ew

    @staticmethod
    def vee_map(M: torch.Tensor) -> torch.Tensor:
        """
        Map a skew-symmetric matrix of shape ``(..., 3, 3)`` to its 3-vector
        ``[M[2,1], M[0,2], M[1,0]]``.
        """
        return torch.stack(
            [
                M[..., 2, 1],
                M[..., 0, 2],
                M[..., 1, 0],
            ],
            dim=-1,
        )

    def _build_obs_idx(self, device) -> torch.LongTensor:
        """
        Build 0-based residue indices, one per experiment. Each ``exp.name``
        (or ``exp.resid``) is expected to encode the residue number.
        """

        def infer_resid(exp_obj) -> int:
            resid_val = getattr(exp_obj, "resid", None)
            if resid_val is not None:
                return int(resid_val)
            name_val = getattr(exp_obj, "name", None)
            if isinstance(name_val, str) and len(name_val) > 0:
                token = name_val.split()[0]
                if token.isdigit():
                    return int(token)
            rep = str(exp_obj)
            match = re.search(r"(\d+)\s*$", rep)
            if match:
                return int(match.group(1))
            raise ValueError(
                f"Cannot infer residue index for experiment {exp_obj!r}. "
                "Expected .resid, .name starting with a number, or repr "
                "ending in a number."
            )

        idxs = [infer_resid(exp) - 1 for exp in self._experimental_data]
        return torch.tensor(idxs, dtype=torch.long, device=device)
