"""Trainer for the BioEmu protein benchmark.

:class:`BioEmuTrainer` extends :class:`mew_og.training.MewOGTrainer`, reusing
its parameter-space initialization, parameter updates, and optimization loop,
while overriding the sampling and loss so they operate on protein ``(r, Q)``
samples produced by :func:`mew_og.benchmark.sampling.generate_batch`.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from mew_og.benchmark.sampling import generate_batch
from mew_og.training.train_mew_og import MewOGTrainer


def _num_iters(n_samples: int, batch_size: int) -> int:
    return max(1, (n_samples + batch_size - 1) // batch_size)


class BioEmuTrainer(MewOGTrainer):
    """MEW-OG trainer for BioEmu protein samples."""

    def __init__(
        self,
        score_model,
        sequence,
        sdes,
        batch_size,
        n_training_samples,
        seed,
        denoiser,
        cache_embeds_dir,
        msa_file,
        msa_host_url,
        augmenter,
        config,
        output_dir,
        device,
        ground_truth_trajectory=None,
        biased_trajectory=None,
    ):
        self.augmenter = augmenter
        model_proxy = SimpleNamespace(augmenter=augmenter)
        super().__init__(
            model=model_proxy,
            config=config,
            output_dir=output_dir,
            ground_truth_trajectory=ground_truth_trajectory,
            biased_trajectory=biased_trajectory,
            device=device,
        )

        self.score_model = score_model
        self.sequence = sequence
        self.sdes = sdes
        self.batch_size = batch_size
        self.n_training_samples = n_training_samples
        self.seed = seed
        self.denoiser = denoiser
        self.cache_embeds_dir = cache_embeds_dir
        self.msa_file = msa_file
        self.msa_host_url = msa_host_url

        self.observables_msm = self._set_observables_msm()
        self.observables_exp = self._set_observables_exp()
        self.best_loss = float("inf")
        self.save_config()

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def iter_samples(self, n_samples: int):
        """Yield ``(r, Q, excess_work)`` per batch (streaming, memory-efficient)."""
        B = int(self.batch_size)
        n_iters = _num_iters(n_samples, B)
        self.augmenter.excess_work.zero_()

        with torch.no_grad():
            for i in tqdm(range(n_iters), desc="Accumulating samples"):
                batch = generate_batch(
                    score_model=self.score_model,
                    sequence=self.sequence,
                    sdes=self.sdes,
                    batch_size=B,
                    seed=self.seed + i if self.seed is not None else i,
                    denoiser=self.denoiser,
                    cache_embeds_dir=self.cache_embeds_dir,
                    msa_file=self.msa_file,
                    msa_host_url=self.msa_host_url,
                    augmenter=self.augmenter,
                )

                r = batch["pos"].to(self.device, non_blocking=True)
                Q = batch["node_orientations"].to(self.device, non_blocking=True)

                excess_work = float(self.augmenter.excess_work)
                if not np.isfinite(excess_work):
                    print(f"Warning: Non-finite excess work {excess_work}, using 0.0")
                    excess_work = 0.0

                yield r, Q, excess_work

                del batch, r, Q

    def sample(self, n_samples: int) -> dict:
        """Collect all ``(r, Q)`` samples on CPU with the mean excess work."""
        r_list, Q_list, excess_works = [], [], []
        for r, Q, ew in self.iter_samples(n_samples):
            r_list.append(r.detach().to("cpu"))
            Q_list.append(Q.detach().to("cpu"))
            excess_works.append(ew)

        r_all = torch.cat(r_list, dim=0)
        Q_all = torch.cat(Q_list, dim=0)
        mean_excess_work = float(sum(excess_works) / max(1, len(excess_works)))

        if not np.isfinite(mean_excess_work):
            print(f"Warning: Non-finite mean excess work {mean_excess_work}, using 0.0")
            mean_excess_work = 0.0

        return {
            "pos": r_all,
            "node_orientations": Q_all,
            "mean_excess_work": mean_excess_work,
        }

    # ------------------------------------------------------------------ #
    # Objective / loss
    # ------------------------------------------------------------------ #
    def _objective(self, params) -> float:
        """Update scaling parameters and return the total loss (for the optimizer)."""
        self._update_parameters(params)
        loss = self.loss_fn()
        loss_value = float(loss)
        self.loss_history.append(loss_value)
        if not np.isfinite(loss_value):
            return 1e3
        return loss_value

    def loss_fn(self) -> torch.Tensor:
        """Compute MSE + gamma * excess-work without keeping samples in memory."""
        n_samples = int(self.config.get("n_training_samples", self.n_training_samples))
        gamma = float(self.config.get("gamma", 1e-4))

        mse_sum = 0.0
        n_obs_total = 0
        excess_work_sum = 0.0

        with torch.no_grad():
            for i, (r, Q, ew) in enumerate(self.iter_samples(n_samples)):
                try:
                    pred, exp = self.calculate_observables(samples=(r, Q))

                    if torch.isnan(pred).any() or torch.isnan(exp).any():
                        print(
                            f"Warning: NaN in observables - pred: {pred}, exp: {exp}"
                        )
                        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
                        exp = torch.where(torch.isnan(exp), torch.zeros_like(exp), exp)

                    diff = pred - exp
                    mse_sum += float((diff * diff).sum().item())
                    n_obs_total += int(diff.numel())
                    excess_work_sum += ew

                    del r, Q, pred, exp, diff
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    continue

        mse_loss = mse_sum / max(1, n_obs_total)

        if not np.isfinite(excess_work_sum):
            print(f"Warning: Non-finite excess work sum {excess_work_sum}, using 0.0")
            excess_work_sum = 0.0

        reg = gamma * (excess_work_sum / _num_iters(n_samples, int(self.batch_size)))
        total_loss = mse_loss + reg

        if not np.isfinite(total_loss):
            print(f"Warning: Non-finite total loss {total_loss}, returning penalty")
            return torch.tensor(1e3, device=self.device, dtype=torch.float32)

        print(
            f"Losses - mse: {mse_loss:.6e}, excess-work: {reg:.6e}, "
            f"total: {total_loss:.6e}"
        )

        if total_loss < self.best_loss:
            previous_best = self.best_loss
            self.best_loss = total_loss
            print(
                f"New best loss: {total_loss:.6e} (previous best: {previous_best:.6e})"
            )
            self.save_params()
        else:
            print(
                f"Current loss: {total_loss:.6e} "
                f"(best so far: {self.best_loss:.6e}) - not saving parameters"
            )

        return torch.tensor(total_loss, device=self.device, dtype=torch.float32)

    def calculate_observables(self, samples):
        """Compute obs-index-filtered predicted expectations vs. experimental."""
        full_obs = self.augmenter.observables_function(*samples)

        if torch.isnan(full_obs).any():
            full_obs = torch.where(
                torch.isnan(full_obs), torch.zeros_like(full_obs), full_obs
            )

        observables_per_sample = full_obs.index_select(
            dim=1, index=self.augmenter.obs_idx
        )
        if torch.isnan(observables_per_sample).any():
            observables_per_sample = torch.where(
                torch.isnan(observables_per_sample),
                torch.zeros_like(observables_per_sample),
                observables_per_sample,
            )

        observables_pred = self.augmenter.predict_expectations(observables_per_sample)
        if torch.isnan(observables_pred).any():
            observables_pred = torch.where(
                torch.isnan(observables_pred),
                torch.zeros_like(observables_pred),
                observables_pred,
            )

        return observables_pred, self.observables_exp.detach()

    # ------------------------------------------------------------------ #
    # Evaluation & persistence
    # ------------------------------------------------------------------ #
    def evaluate(self, n_samples: int = 5120, show: bool = False):
        """Sample a large batch, save it, and report predicted vs. experimental."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        batch = self.sample(n_samples=n_samples)
        r = batch["pos"].to(self.device)
        Q = batch["node_orientations"].to(self.device)

        self.save_dataset(
            batch,
            self.output_dir / _random_name("batch_", 4),
        )
        self.save_params()

        observables_pred, observables_exp = self.calculate_observables(samples=(r, Q))
        observables_pred = observables_pred.detach().cpu().squeeze()
        observables_exp = observables_exp.detach().cpu().squeeze()

        plt.plot(observables_exp, observables_exp, "k--", label="Experimental")
        plt.plot(observables_exp, observables_pred, "o", label="Guided", color="C0")
        plt.legend()
        plt.savefig(self.output_dir / "observables_comparison.pdf")
        if show:
            plt.show()
        plt.close()

        return observables_pred, observables_exp

    def save_dataset(self, batch, npz_path):
        data = {
            "pos": batch["pos"].cpu().numpy(),
            "node_orientations": batch["node_orientations"].cpu().numpy(),
        }
        np.savez(npz_path, **data, sequence=self.sequence)

    def save_config(self):
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(_json_safe(self.config), f, indent=2)

    def save_params(self):
        params_dict = {
            "best_loss": float(self.best_loss),
            "loss_info": {
                "current_best": float(self.best_loss),
                "timestamp": datetime.now().isoformat(),
            },
        }
        scaling_function = self.augmenter.scaling_function
        params_list = []
        for func in scaling_function:
            func_params = {}
            if hasattr(func, "_b"):
                b_param = getattr(func, "_b")
                if isinstance(b_param, torch.Tensor):
                    func_params["b"] = round(b_param.detach().item(), 6)
                else:
                    func_params["b"] = round(float(b_param), 6)
                if hasattr(func, "a"):
                    func_params["a_calculated"] = round(func.a.detach().item(), 6)
            params_list.append(func_params)
        params_dict["scaling_function_params"] = params_list

        with open(self.output_dir / "params.json", "w") as f:
            json.dump(params_dict, f, indent=4)

    def _set_observables_msm(self) -> torch.Tensor:
        msm_tensor = (
            torch.tensor(
                [exp.observables_msm for exp in self.augmenter.experimental_data]
            )
            .squeeze()
            .to(self.device)
        )
        if torch.isnan(msm_tensor).any():
            msm_tensor = torch.where(
                torch.isnan(msm_tensor), torch.zeros_like(msm_tensor), msm_tensor
            )
        return msm_tensor

    def _set_observables_exp(self) -> torch.Tensor:
        exp_tensor = (
            torch.tensor(
                [exp.observables_exp for exp in self.augmenter.experimental_data]
            )
            .squeeze()
            .to(self.device)
        )
        if torch.isnan(exp_tensor).any():
            exp_tensor = torch.where(
                torch.isnan(exp_tensor), torch.zeros_like(exp_tensor), exp_tensor
            )
        return exp_tensor


def _random_name(prefix: str = "run", length: int = 4) -> str:
    import random
    import string

    rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=length))
    return f"{prefix}{rand_str}"


def _json_safe(obj):
    """Best-effort conversion of a config to JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
