"""BioEmu sampling glue for observable-guided protein generation.

This module contains the SO(3)/Euclidean diffusion machinery that bridges the
external `bioemu` package with the MEW-OG guidance augmenter. It implements the
DPM-style solver that injects the guidance force at every step, plus helpers to
build context graphs, generate batches, and serialize samples to PDB/XTC.

The `bioemu` package is a third-party dependency and is imported directly.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import mdtraj
import numpy as np
import torch
from torch_geometric.data.batch import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.convert_chemgraph import (
    filter_unphysical_traj,
    get_atom37_from_frames,
)
from bioemu.denoiser import (
    EulerMaruyamaPredictor,
    _t_from_lambda,
    get_score,
)
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.openfold.np.protein import Protein, to_pdb
from bioemu.sde_lib import CosineVPSDE, SDE
from bioemu.so3_sde import SO3SDE, apply_rotvec_to_rotmat

from mew_og.benchmark.augmenter import BioEmuAugmenter

logger = logging.getLogger(__name__)


def calculate_bias(r, Q, augmenter, t):
    """Return the guidance force ``(hr_t, hq_t)`` for the batch, or zeros."""
    if augmenter is None:
        return torch.zeros_like(r), torch.zeros_like(r)
    hr_t, hq_t = augmenter((r, Q), t)
    return hr_t, hq_t


def detach_score(score):
    """Detach a score dict to avoid tracking gradients."""
    return {k: v.detach() for k, v in score.items()}


def _hat_map(rotvec: torch.Tensor) -> torch.Tensor:
    """Map (..., 3) rotvec -> (..., 3, 3) skew-symmetric matrix (so(3))."""
    x = rotvec[..., 0]
    y = rotvec[..., 1]
    z = rotvec[..., 2]
    zero = torch.zeros_like(x)
    M = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    return M


def _project_to_so3(mat: torch.Tensor) -> torch.Tensor:
    """Project a batch of 3x3 matrices to SO(3) by SVD with det correction."""
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    UVt = U @ Vh
    det = torch.det(UVt)
    ones = torch.ones_like(det)
    D = torch.stack([ones, ones, det], dim=-1)
    Dmat = torch.diag_embed(D)
    Q = U @ Dmat @ Vh
    return Q


def add_bias(score, sde, batch, augmenter, t):
    """Tweedie posterior mean for positions/rotations, then add guidance."""
    alpha_t, sigma_t = sde.mean_coeff_and_std(batch.pos, t, batch.batch)
    expand_dims = (1,) * (batch.pos.dim() - alpha_t.dim())
    alpha_exp = alpha_t.view(*alpha_t.shape, *expand_dims)
    sigma2_exp = (sigma_t**2).view(*sigma_t.shape, *expand_dims)

    r0_hat = (batch.pos + sigma2_exp * score["pos"]) / alpha_exp

    omega = score["node_orientations"]
    hat_omega = _hat_map(omega)

    alpha_3x3 = alpha_exp.unsqueeze(-1)
    sigma2_3x3 = sigma2_exp.unsqueeze(-1)

    Q_t = batch.node_orientations
    tilde_Q0 = (Q_t + sigma2_3x3 * hat_omega) / alpha_3x3
    Q0_hat = _project_to_so3(tilde_Q0)

    hr_t, hq_t = calculate_bias(r=r0_hat, Q=Q0_hat, augmenter=augmenter, t=t)

    score["pos"] = score["pos"] + hr_t
    score["node_orientations"] = score["node_orientations"] + hq_t
    return score


def _broadcast_to_target(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Right-pad singleton dims onto ``x`` until it broadcasts to ``target``."""
    x = x.clone()
    while x.dim() < target.dim():
        x = x.unsqueeze(-1)
    return x.expand_as(target)


def add_bias_manifold(score, pos_sde, so3_sde, batch, augmenter, t):
    """Add guidance on the SO(3) x R^3 manifold at the clean estimate."""
    alpha_pos_t, sigma_pos_t = pos_sde.mean_coeff_and_std(batch.pos, t, batch.batch)
    alpha_pos = _broadcast_to_target(alpha_pos_t, batch.pos)
    sigma2_pos = _broadcast_to_target(sigma_pos_t**2, batch.pos)
    r0_hat = (batch.pos + sigma2_pos * score["pos"]) / alpha_pos

    alpha_rot_t, sigma_rot_t = so3_sde.mean_coeff_and_std(
        batch.node_orientations, t, batch.batch
    )
    alpha_rot = _broadcast_to_target(alpha_rot_t, score["node_orientations"])
    sigma2_rot = _broadcast_to_target(sigma_rot_t**2, score["node_orientations"])
    omega = score["node_orientations"]
    rot_update = (sigma2_rot / alpha_rot) * omega
    Q0_hat = apply_rotvec_to_rotmat(batch.node_orientations, rot_update)

    hr_t, hq_t = calculate_bias(r=r0_hat, Q=Q0_hat, augmenter=augmenter, t=t)

    score["pos"] = score["pos"] + hr_t / _broadcast_to_target(alpha_pos_t, hr_t)
    score["node_orientations"] = score["node_orientations"] + hq_t / alpha_rot
    return score


def dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set = set(),
    noise: float = 0.0,
    augmenter: Optional[BioEmuAugmenter] = None,
) -> ChemGraph:
    """
    DPM solver for the VPSDE (cosine schedule) that tracks gradients w.r.t.
    positions and node orientations so the guidance force can be applied.
    """
    assert isinstance(batch, ChemGraph)
    assert max_t < 1.0

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)

    pos_sde = sdes["pos"]
    so3_sde = sdes["node_orientations"]
    assert isinstance(pos_sde, CosineVPSDE)
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    batch = batch.replace(
        pos=pos_sde.prior_sampling(batch.pos.shape, device=device),
        node_orientations=so3_sde.prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    batch.pos.requires_grad_()
    batch.node_orientations.requires_grad_()

    torch.set_grad_enabled(False)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1), device=device)
    ts_min, ts_max = 0.0, 1.0

    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }

    for i in range(N - 1):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_hat = t - noise * dt if (i > 0 and ts_min < t[0] < ts_max) else t

        vals_hat = {}
        for name, predictor in noisers.items():
            vals_hat[name] = predictor.forward_sde_step(
                x=batch[name], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)
        batch_hat = batch_hat.replace(
            pos=batch_hat.pos.detach().requires_grad_(),
            node_orientations=batch_hat.node_orientations.detach().requires_grad_(),
        )

        with torch.enable_grad():
            score = get_score(
                batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes
            )
            score = add_bias_manifold(
                score, pos_sde, so3_sde, batch_hat, augmenter, t_hat
            )
        score = detach_score(score)

        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(batch.pos, t_hat, batch.batch)
        alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
            batch.pos, t + dt, batch.batch
        )
        lambda_t = torch.log(alpha_t / sigma_t)
        lambda_t_next = torch.log(alpha_t_next / sigma_t_next)
        h_t = lambda_t_next - lambda_t

        t_lambda = _t_from_lambda(pos_sde, (lambda_t + lambda_t_next) / 2)
        t_lambda = torch.full((batch.num_graphs,), t_lambda[0][0], device=device)
        alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
            batch.pos, t_lambda, batch.batch
        )

        u = (
            alpha_t_lambda / alpha_t * batch_hat.pos
            + sigma_t_lambda * sigma_t * (torch.exp(h_t / 2) - 1) * score["pos"]
        )
        batch_u = batch.replace(pos=u)

        so3_pred = EulerMaruyamaPredictor(
            corruption=so3_sde,
            noise_weight=0.0,
            marginal_concentration_factor=1.0,
        )
        drift, _ = so3_pred.reverse_drift_and_diffusion(
            x=batch_hat.node_orientations,
            score=score["node_orientations"],
            t=t_hat,
            batch_idx=batch.batch,
        )
        sample, _ = so3_pred.update_given_drift_and_diffusion(
            x=batch_hat.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=(t_lambda - t_hat)[0],
        )
        batch_u = batch_u.replace(node_orientations=sample)

        batch_u = batch_u.replace(
            pos=batch_u.pos.detach().requires_grad_(),
            node_orientations=batch_u.node_orientations.detach().requires_grad_(),
        )

        with torch.enable_grad():
            score_u = get_score(
                batch=batch_u, t=t_lambda, sdes=sdes, score_model=score_model
            )
            score_u = add_bias_manifold(
                score_u, pos_sde, so3_sde, batch_u, augmenter, t_lambda
            )
        score_u = detach_score(score_u)

        pos_next = (
            alpha_t_next / alpha_t * batch_hat.pos
            + sigma_t_next * sigma_t_lambda * (torch.exp(h_t) - 1) * score_u["pos"]
        )
        batch_next = batch.replace(pos=pos_next)

        dt_hat = (t + dt - t_hat)[0]
        node_score = (
            score_u["node_orientations"]
            + 0.5
            * (score_u["node_orientations"] - score["node_orientations"])
            / ((t_lambda - t_hat)[0])
            * dt_hat
        )
        drift2, _ = so3_pred.reverse_drift_and_diffusion(
            x=batch_u.node_orientations,
            score=node_score,
            t=t_lambda,
            batch_idx=batch.batch,
        )
        sample2, _ = so3_pred.update_given_drift_and_diffusion(
            x=batch_hat.node_orientations,
            drift=drift2,
            diffusion=0.0,
            dt=dt_hat,
        )
        batch = batch_next.replace(node_orientations=sample2)
        batch["pos"] = batch.pos.detach()
        batch["node_orientations"] = batch.node_orientations.detach()

    return batch


def get_context_chemgraph(
    sequence: str,
    cache_embeds_dir: Optional[str] = None,
    msa_file: Optional[str] = None,
    msa_host_url: Optional[str] = None,
) -> ChemGraph:
    """Build a context ChemGraph (embeddings + edges) for a sequence."""
    n = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    single_embeds = torch.from_numpy(np.load(single_embeds_file))
    pair_embeds = torch.from_numpy(np.load(pair_embeds_file))
    assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
    assert single_embeds.shape[0] == n
    assert len(single_embeds.shape) == 2
    _, _, n_pair_feats = pair_embeds.shape

    pair_embeds = pair_embeds.view(n**2, n_pair_feats)

    edge_index = torch.cat(
        [
            torch.arange(n).repeat_interleave(n).view(1, n**2),
            torch.arange(n).repeat(n).view(1, n**2),
        ],
        dim=0,
    )
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))

    return ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
        sequence=sequence,
    )


def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    cache_embeds_dir: Optional[str],
    msa_file: Optional[str] = None,
    msa_host_url: Optional[str] = None,
    augmenter: Optional[BioEmuAugmenter] = None,
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples, using GPU if available.

    Args:
        score_model: Score model.
        sequence: Amino acid sequence.
        sdes: SDEs defining the corruption process (keys 'pos',
            'node_orientations').
        batch_size: Batch size.
        seed: Random seed.
        denoiser: Callable sampler (e.g. a partial of :func:`dpm_solver`).
        cache_embeds_dir: Directory to store MSA embeddings.
        msa_file: Optional path to an MSA A3M file.
        msa_host_url: MSA server URL for colabfold.
        augmenter: Augmenter for guided sampling; ``None`` disables guidance.
    """
    context_chemgraph = get_context_chemgraph(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampled_chemgraph_batch = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
        augmenter=augmenter,
    )
    assert isinstance(sampled_chemgraph_batch, Batch)
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
    node_orientations = torch.stack(
        [x.node_orientations for x in sampled_chemgraphs]
    ).to("cpu")

    return {"pos": pos, "node_orientations": node_orientations}


def save_pdb_and_xtc(
    pos_nm: torch.Tensor,
    node_orientations: torch.Tensor,
    sequence: str,
    topology_path,
    xtc_path,
    filter_samples: bool = True,
) -> None:
    """
    Convert coarse-grained structures to backbone atoms and save the first
    frame as a PDB and all frames to an XTC trajectory. Load them back with
    ``mdtraj.load_xtc(xtc_path, top=topology_path)``.

    Args:
        pos_nm: (batch_size, N, 3) positions in nm.
        node_orientations: (batch_size, N, 3, 3) node orientations.
        sequence: Amino acid sequence.
        topology_path: Path to save the PDB file.
        xtc_path: Path to save the XTC trajectory file.
        filter_samples: Filter out unphysical samples (long bonds, clashes).
    """
    batch_size, _, _ = pos_nm.shape
    assert pos_nm.shape == (batch_size, len(sequence), 3)
    assert node_orientations.shape == (batch_size, len(sequence), 3, 3)

    # PDB files store Angstroms while mdtraj.Trajectory stores nm. Some
    # upstream generators already emit Angstrom-like coordinates, so infer a
    # reasonable scale based on magnitude to avoid overflowing PDB columns.
    with torch.no_grad():
        max_abs = torch.as_tensor(pos_nm).abs().max().item()
    scale_to_angstrom = 1.0 if max_abs > 100.0 else 10.0
    pos_angstrom = pos_nm * scale_to_angstrom

    with torch.no_grad():
        diffs = pos_angstrom[0, 1:, :] - pos_angstrom[0, :-1, :]
        dists = torch.linalg.norm(diffs, dim=-1)
        median_dist = torch.median(dists).item() if dists.numel() > 0 else None
        if median_dist is not None and np.isfinite(median_dist) and median_dist > 0:
            target_dist = 3.8  # A
            scale_norm = float(target_dist / median_dist)
            scale_norm = float(np.clip(scale_norm, 0.01, 100.0))
            pos_angstrom = pos_angstrom * scale_norm

    pos_angstrom = pos_angstrom - pos_angstrom.mean(axis=1, keepdims=True)

    _local_write_pdb(
        pos=pos_angstrom[0],
        node_orientations=node_orientations[0],
        sequence=sequence,
        filename=topology_path,
    )

    xyz_angstrom = []
    for i in range(batch_size):
        atom_37, atom_37_mask, _ = get_atom37_from_frames(
            pos=pos_angstrom[i],
            node_orientations=node_orientations[i],
            sequence=sequence,
        )
        xyz_angstrom.append(atom_37.view(-1, 3)[atom_37_mask.flatten()].cpu().numpy())

    topology = mdtraj.load_topology(topology_path)

    traj = mdtraj.Trajectory(xyz=np.stack(xyz_angstrom) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        logger.info("Filtering samples ...")
        traj = filter_unphysical_traj(traj)
        logger.info(
            f"Filtered {num_samples_unfiltered} samples down to {len(traj)} "
            "based on structure criteria. Filtering can be disabled with "
            "`filter_samples=False`."
        )

    traj.superpose(reference=traj, frame=0)
    traj.save_xtc(xtc_path)


def _local_write_pdb(
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    sequence: str,
    filename,
) -> None:
    """
    Like ``bioemu.convert_chemgraph._write_pdb`` but with residue_index
    starting at 1 (avoids 0-based residue numbering in the output PDB).
    """
    assert len(pos.shape) == 2
    num_residues = pos.shape[0]

    atom_37, atom_37_mask, aatype = get_atom37_from_frames(
        pos=pos, node_orientations=node_orientations, sequence=sequence
    )

    protein = Protein(
        atom_positions=atom_37.cpu().numpy(),
        aatype=aatype.cpu().numpy(),
        atom_mask=atom_37_mask.cpu().numpy(),
        residue_index=np.arange(1, num_residues + 1, dtype=np.int64),
        b_factors=np.zeros((num_residues, 37)),
    )
    with open(filename, "w") as f:
        f.write(to_pdb(protein))
