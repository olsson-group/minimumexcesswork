"""MEW-OG trainer for observable-guided sampling."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.optimize import minimize

from mew_og.guidance.augmenter import Augmenter
from mew_og.guidance.mew_og_model import MewOGModel
from mew_og.guidance.scaling import ExponentialScaling
from mew_og.io.hdf5 import write_hdf5
from mew_og.utils.tensor import filter_outliers


class MewOGTrainer:
    """
    Trainer for MEW-OG (Minimum-Excess-Work Observable Guidance).

    This trainer optimizes the scaling function parameters to minimize
    the discrepancy between generated samples and experimental observables.

    Parameters
    ----------
    model : MewOGModel
        The MEW-OG model to train.
    config : dict
        Training configuration.
    output_dir : str or Path
        Directory for saving outputs.
    ground_truth_trajectory : torch.Tensor, optional
        Ground truth samples for evaluation.
    biased_trajectory : torch.Tensor, optional
        Biased samples for comparison.
    device : str or torch.device
        Device for computations.
    """

    def __init__(
        self,
        model: MewOGModel,
        config: dict,
        output_dir: Union[str, Path],
        ground_truth_trajectory: Optional[torch.Tensor] = None,
        biased_trajectory: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.ground_truth_trajectory = ground_truth_trajectory
        self.biased_trajectory = biased_trajectory

        # Training parameters
        self.n_training_samples = config.get("n_training_samples", 1000)
        self.gamma = config.get("gamma", 1e-4)  # Regularization weight
        self.outlier_threshold = config.get("outlier_threshold", 10.0)
        self.outlier_num_std = config.get("outlier_num_std", 3)

        # Optimization parameters
        self.param_bounds = config.get("params", {})
        self.param_names = []
        self.loss_history = []
        self.seed = None

    def train(
        self,
        kind: str = "bayesian-optimization",
        n_calls: int = 50,
        seed: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Train the MEW-OG model.

        Parameters
        ----------
        kind : str
            Optimization method ('bayesian-optimization', 'grid-search', or 'minimize').
        n_calls : int
            Number of optimization iterations (for BO).
        seed : int, optional
            Random seed.
        **kwargs
            Additional arguments for the optimizer.

        Returns
        -------
        dict
            Optimization result.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
        self.seed = seed

        if kind == "bayesian-optimization":
            return self._train_bayesian_optimization(n_calls=n_calls, seed=seed, **kwargs)
        elif kind == "grid-search":
            return self._train_grid_search(**kwargs)
        elif kind == "minimize":
            return self._train_minimize(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {kind}")

    def _train_bayesian_optimization(
        self,
        n_calls: int = 50,
        seed: int = 42,
        threshold: float = 1e-2,
        n_best: int = 2,
        **kwargs,
    ) -> dict:
        """
        Train using Bayesian optimization.

        Note: This is a simplified grid search fallback if skopt is not available.
        """
        # Initialize parameter space
        param_space = self._initialize_parameters()

        best_loss = np.inf
        best_params = None

        print(f"Running optimization with {n_calls} iterations...")

        for i in range(n_calls):
            # Random sample from parameter space
            params = []
            for bounds in param_space:
                val = np.random.uniform(bounds[0], bounds[1])
                params.append(val)

            # Evaluate
            loss = self._objective(params)

            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                print(f"  Iteration {i + 1}: New best loss = {loss:.6f}")

            if loss < threshold:
                print(f"  Converged at iteration {i + 1}")
                break

        # Set best parameters
        if best_params is not None:
            self._update_parameters(best_params)

        return {
            "best_params": best_params,
            "best_loss": best_loss,
            "n_iterations": i + 1,
        }

    def _train_minimize(self, **kwargs) -> dict:
        """Train using scipy.optimize.minimize."""
        param_space = self._initialize_parameters()
        x0 = [(b[0] + b[1]) / 2 for b in param_space]
        bounds = param_space

        result = minimize(
            self._objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "disp": True},
        )

        self._update_parameters(result.x)

        return {
            "best_params": result.x,
            "best_loss": result.fun,
            "success": result.success,
        }

    def _train_grid_search(self, n_points: int = 10, **kwargs) -> dict:
        """Train using grid search."""
        param_space = self._initialize_parameters()

        # Create grid
        grids = []
        for bounds in param_space:
            grids.append(np.linspace(bounds[0], bounds[1], n_points))

        best_loss = np.inf
        best_params = None

        # Evaluate all combinations
        from itertools import product
        for params in product(*grids):
            loss = self._objective(list(params))
            if loss < best_loss:
                best_loss = loss
                best_params = list(params)

        if best_params is not None:
            self._update_parameters(best_params)

        return {
            "best_params": best_params,
            "best_loss": best_loss,
        }

    def _initialize_parameters(self) -> List[tuple]:
        """
        Initialize parameter bounds from config.

        Returns
        -------
        list of tuple
            List of (low, high) bounds for each parameter.
        """
        param_space = []
        self.param_names = []

        for param_name, bounds in self.param_bounds.items():
            # Extract base parameter name
            base_name = param_name.split(".")[-1] if "." in param_name else param_name
            base_name = base_name.lstrip("_")

            # Create parameter for each scaling function
            for i in range(len(self.model.augmenter.scaling_function)):
                self.param_names.append(f"{base_name}{i}")
                param_space.append((bounds[0], bounds[1]))

        return param_space

    def _update_parameters(self, params: List[float]) -> None:
        """
        Update scaling function parameters.

        Parameters
        ----------
        params : list of float
            Parameter values.
        """
        param_dict = dict(zip(self.param_names, params))

        with torch.no_grad():
            for param_name, value in param_dict.items():
                # Extract base name and index
                base_name = "".join(c for c in param_name if not c.isdigit())
                idx = int("".join(c for c in param_name if c.isdigit()))

                # Set parameter
                sf = self.model.augmenter.scaling_function[idx]
                param_tensor = getattr(sf, f"_{base_name}", None)
                if param_tensor is not None:
                    param_tensor.copy_(torch.tensor([value]))

    def _objective(self, params: List[float]) -> float:
        """
        Compute the objective function.

        Parameters
        ----------
        params : list of float
            Current parameter values.

        Returns
        -------
        float
            Loss value.
        """
        self._update_parameters(params)

        try:
            # Generate samples
            samples = self.model.sample(
                n_samples=self.n_training_samples,
                seed=self.seed,
            )
            samples = samples.view(samples.shape[0], -1).cpu()

            # Filter outliers
            samples = filter_outliers(
                samples,
                threshold=self.outlier_threshold,
                num_std_dev=self.outlier_num_std,
            )

            # Compute observables
            obs_per_sample, obs_exp = self.model.augmenter.transform(
                samples, return_experimental=True
            )
            obs_pred = self.model.augmenter.predict_expectations(obs_per_sample)

            # MSE loss
            mse_loss = torch.mean((obs_pred - obs_exp) ** 2).item()

            # Regularization (excess work)
            excess_work = self.model.sampler.excess_work.item()
            total_loss = mse_loss + self.gamma * excess_work

            self.loss_history.append(total_loss)

            if not np.isfinite(total_loss):
                return 1e6

            return total_loss

        except Exception as e:
            print(f"Error in objective: {e}")
            return 1e6

    def evaluate(
        self,
        n_samples: int = 10000,
        save_results: bool = True,
        group_name: str = "0",
    ) -> dict:
        """
        Evaluate the trained model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        save_results : bool
            Whether to save results to HDF5.
        group_name : str
            Group name for HDF5 storage.

        Returns
        -------
        dict
            Evaluation results.
        """
        # Generate samples
        all_samples = self.model.sample(
            n_samples=n_samples,
            return_all_samples=True,
            seed=self.seed,
        )
        samples = all_samples[-1].view(-1, 1)

        # Compute observables
        obs_per_sample, obs_exp = self.model.augmenter.transform(
            samples, return_experimental=True
        )
        obs_pred = self.model.augmenter.predict_expectations(obs_per_sample)

        results = {
            "samples": samples.squeeze(),
            "observables_pred": obs_pred,
            "observables_exp": obs_exp,
            "loss_history": self.loss_history,
        }

        # Add scaling function parameters
        params = self.model.get_scaling_params()
        results["scaling_params"] = params

        # Print summary
        print(f"\nEvaluation Results:")
        print(f"  Observables predicted: {obs_pred.squeeze().tolist()}")
        print(f"  Observables expected:  {obs_exp.squeeze().tolist()}")
        print(f"  MSE: {torch.mean((obs_pred - obs_exp) ** 2).item():.6f}")

        if save_results:
            self._save_results(results, group_name)

        return results

    def _save_results(self, results: dict, group_name: str) -> None:
        """Save results to HDF5."""
        data = {
            "samples": results["samples"],
            "observables_pred": results["observables_pred"].squeeze(),
            "observables_exp": results["observables_exp"].squeeze(),
        }

        write_hdf5(
            self.output_dir / f"results-{group_name}.h5",
            data,
            group_name=group_name,
        )

