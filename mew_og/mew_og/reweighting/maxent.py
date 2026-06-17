"""Maximum entropy reweighting estimator."""

import pickle
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from scipy.optimize import minimize


class MaxEntReweightingEstimator(torch.nn.Module):
    """
    Maximum entropy reweighting estimator.

    This class implements the maximum entropy principle to reweight samples
    from a biased distribution to match experimental observables. It finds
    Lagrange multipliers (lambdas) that minimize the KL divergence from the
    original distribution while satisfying the constraints.

    The reweighted distribution is:
        w(x) ∝ exp(-sum_f lambda_f * f(x))

    where f(x) are the observable functions.

    Parameters
    ----------
    experimental_data : list of Experiment, optional
        List of experiments containing observable functions and target values.
    lambdas : torch.Tensor, optional
        Initial Lagrange multipliers.
    device : str or torch.device
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.
    """

    def __init__(
        self,
        experimental_data: Optional[List] = None,
        lambdas: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        # Trajectory data (set during fit)
        self._trajectory_data = None

        # Experimental data and observables
        self._experimental_data = None
        self.observables_function = None
        self.observables_exp = None
        self.observables_exp_uncertainty = None
        self.observables_per_sample = None

        # Weights
        self.register_buffer("w", None)
        self.register_buffer("w_initial", None)

        # Lambdas (Lagrange multipliers)
        self._lambdas = lambdas

        if experimental_data is not None:
            self.experimental_data = experimental_data

    @property
    def lambdas(self) -> Optional[torch.Tensor]:
        """Get Lagrange multipliers."""
        return self._lambdas

    @lambdas.setter
    def lambdas(self, value: Optional[torch.Tensor]) -> None:
        """Set Lagrange multipliers."""
        if value is not None:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            self._lambdas = value.to(self.dtype).to(self.device)
        else:
            self._lambdas = None

    @property
    def experimental_data(self) -> Optional[List]:
        """Get experimental data."""
        return self._experimental_data

    @experimental_data.setter
    def experimental_data(self, value: List) -> None:
        """Set experimental data and initialize observables."""
        self._experimental_data = value
        self._initialize_experimental_observables()
        self._initialize_lambdas()

    @property
    def trajectory_data(self) -> Optional[torch.Tensor]:
        """Get trajectory data."""
        return self._trajectory_data

    @trajectory_data.setter
    def trajectory_data(self, value: torch.Tensor) -> None:
        """Set trajectory data and initialize weights and observables."""
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self._trajectory_data = value.to(self.dtype).to(self.device)
        self._initialize_weights()
        self._initialize_observables_per_sample()

    def _initialize_experimental_observables(self) -> None:
        """Initialize observables from experimental data."""
        if self._experimental_data is None:
            return

        # Extract observable function
        if len(set(exp.observables_function for exp in self._experimental_data)) == 1:
            self.observables_function = self._experimental_data[0].observables_function
        else:
            # Multiple different observable functions
            def combined_obs(x):
                return torch.hstack([exp.observables_function(x) for exp in self._experimental_data])
            self.observables_function = combined_obs

        # Extract experimental values
        self.observables_exp = torch.vstack([
            exp.observables_exp for exp in self._experimental_data
        ]).T.to(self.device)

        self.observables_exp_uncertainty = torch.vstack([
            exp.observables_exp_uncertainty for exp in self._experimental_data
        ]).T.to(self.device)

    def _initialize_lambdas(self) -> None:
        """Initialize Lagrange multipliers randomly."""
        if self._experimental_data is not None and self._lambdas is None:
            n_obs = len(self._experimental_data)
            self._lambdas = (torch.rand(n_obs, dtype=self.dtype, device=self.device) - 0.5)

    def _initialize_weights(self) -> None:
        """Initialize uniform weights."""
        if self._trajectory_data is not None:
            n = len(self._trajectory_data)
            self.w_initial = torch.ones(n, dtype=self.dtype, device=self.device) / n
            self.w = torch.empty_like(self.w_initial)

    def _initialize_observables_per_sample(self) -> None:
        """Compute observables for each sample."""
        if self._trajectory_data is not None and self.observables_function is not None:
            self.observables_per_sample = self.observables_function(self._trajectory_data)

    def fit(
        self,
        x: torch.Tensor,
        max_iter: int = 50,
        ftol: float = 1e-10,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Fit the Lagrange multipliers to match experimental observables.

        Uses L-BFGS-B optimization with multiple random restarts.

        Parameters
        ----------
        x : torch.Tensor
            Trajectory data.
        max_iter : int
            Maximum number of random restarts.
        ftol : float
            Function tolerance for optimizer convergence.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Optimization result containing 'lambdas', 'error', 'success'.
        """
        self.trajectory_data = x

        best_error = np.inf
        best_result = None
        best_lambdas = None

        for iteration in range(max_iter):
            # Generate new random seed for each iteration
            if seed is not None:
                iter_seed = seed + iteration
            else:
                iter_seed = np.random.randint(0, 10000)

            torch.manual_seed(iter_seed)
            np.random.seed(iter_seed)

            # Random initialization
            self._lambdas = (torch.rand(len(self._experimental_data), dtype=self.dtype) - 0.5)

            # Optimize
            result = minimize(
                self._loss_fn,
                self._lambdas.detach().numpy(),
                method="L-BFGS-B",
                options={"ftol": ftol, "maxiter": 10000},
            )

            # Update lambdas with result
            self._lambdas = torch.tensor(result.x, dtype=self.dtype)

            # Compute error
            self.w = self.weights(self._trajectory_data)
            observables_pred = self.predict_observables(self._trajectory_data)
            error = torch.mean((observables_pred - self.observables_exp.squeeze()) ** 2).item()

            if error < best_error:
                best_error = error
                best_result = result
                best_lambdas = self._lambdas.clone()

            if verbose:
                print(f"Iteration {iteration + 1}/{max_iter}: Error = {error:.6f}")

        # Set best result
        self._lambdas = best_lambdas
        self.w = self.weights(self._trajectory_data)

        if verbose:
            print(f"Best error: {best_error:.6f}")

        return {
            "lambdas": best_lambdas,
            "error": best_error,
            "success": best_result.success if best_result else False,
        }

    def _loss_fn(self, lambdas: np.ndarray) -> float:
        """
        Maximum entropy loss function for optimization.

        Parameters
        ----------
        lambdas : np.ndarray
            Current Lagrange multipliers.

        Returns
        -------
        float
            Loss value (negative log partition function + constraint term).
        """
        self._lambdas = torch.tensor(lambdas, dtype=self.dtype)

        # Compute lambda * observables for each sample
        lambda_obs = -torch.sum(self._lambdas * self.observables_per_sample, dim=1, keepdim=True)

        # Log weights
        log_w = lambda_obs + torch.log(self.w_initial.unsqueeze(1))

        # Log partition function
        log_z = torch.logsumexp(log_w, dim=0)

        # Constraint term
        lambda_exp = torch.sum(self._lambdas * self.observables_exp.squeeze())

        loss = log_z + lambda_exp
        return loss.item()

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reweighting factors for samples.

        Parameters
        ----------
        x : torch.Tensor
            Samples.

        Returns
        -------
        torch.Tensor
            Normalized weights.
        """
        observables_per_sample = self.observables_function(x)
        return self._weights(observables_per_sample)

    def _weights(self, observables_per_sample: torch.Tensor) -> torch.Tensor:
        """
        Compute weights from observables.

        Parameters
        ----------
        observables_per_sample : torch.Tensor
            Observable values for each sample.

        Returns
        -------
        torch.Tensor
            Normalized weights.
        """
        lambda_obs = -torch.sum(self._lambdas * observables_per_sample, dim=1, keepdim=True)
        w_initial = torch.ones_like(lambda_obs) / len(lambda_obs)
        log_w = lambda_obs + torch.log(w_initial)
        log_z = torch.logsumexp(log_w, dim=0)
        w = torch.exp(log_w - log_z)
        return w.squeeze()

    def predict_observables(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict observable values under the reweighted distribution.

        Parameters
        ----------
        x : torch.Tensor
            Samples.

        Returns
        -------
        torch.Tensor
            Weighted mean observable values.
        """
        observables_per_sample = self.observables_function(x)
        weights = self._weights(observables_per_sample)
        return torch.matmul(weights, observables_per_sample)

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save the estimator to a file.

        Parameters
        ----------
        file_path : str or Path
            Output file path.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: Union[str, Path]) -> "MaxEntReweightingEstimator":
        """
        Load an estimator from a file.

        Parameters
        ----------
        file_path : str or Path
            Input file path.

        Returns
        -------
        MaxEntReweightingEstimator
            Loaded estimator.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

