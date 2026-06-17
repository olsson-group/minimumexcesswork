"""DDPM trainer for the Prinz potential toy system."""

from pathlib import Path
from typing import Optional, Union

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from mew_og.io.checkpoints import save_checkpoint
from mew_og.models.beta_schedule import LinearBetaScheduler
from mew_og.models.losses import ddpm_loss
from mew_og.models.score_network import ScoreBasedDDPM
from mew_og.samplers.vp_sde import VPSDESampler
from mew_og.utils.tensor import filter_outliers


class DDPMTrainer:
    """
    Trainer for score-based DDPM on toy data.

    Parameters
    ----------
    model : ScoreBasedDDPM
        Score network to train.
    data_loader : DataLoader
        Training data loader.
    config : dict
        Training configuration.
    output_dir : str or Path
        Directory for saving outputs.
    device : str or torch.device
        Device for training.
    ground_truth_trajectory : torch.Tensor, optional
        Ground truth samples for evaluation.
    """

    def __init__(
        self,
        model: ScoreBasedDDPM,
        data_loader,
        config: dict,
        output_dir: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        ground_truth_trajectory: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.ground_truth_trajectory = ground_truth_trajectory

        # Training parameters
        self.n_epochs = config.get("n_epochs", 100)
        self.eval_frequency = config.get("eval_frequency", 10)
        self.lr = config.get("lr", 1e-4)

        # Set up beta scheduler
        beta_kwargs = config.get("beta_scheduler_kwargs", {})
        self.beta_fn = LinearBetaScheduler(device=device, **beta_kwargs)

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Set up sampler for evaluation
        self.sampler = VPSDESampler(
            score_network=self.model,
            beta_fn=self.beta_fn,
            device=device,
        )

        # Training state
        self.loss_history = []
        self.current_epoch = 0

    def train(self) -> None:
        """Run the training loop."""
        print(f"Training for {self.n_epochs} epochs...")

        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            self.current_epoch = epoch
            avg_loss = self._train_epoch()
            self.loss_history.append(avg_loss)

            # Evaluate and save periodically
            if (epoch + 1) % self.eval_frequency == 0:
                print(f"\nEpoch {epoch + 1}: Loss = {avg_loss:.6f}")
                self.evaluate()
                self.save()

        print("Training complete!")
        self.save()

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in self.data_loader:
            x = batch["x"].to(self.device)
            batch_size = x.shape[0]

            # Compute loss
            loss = ddpm_loss(self.model, x, self.beta_fn.integral)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    def evaluate(self, n_samples: int = 5000) -> None:
        """
        Evaluate the model by generating samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        """
        self.model.eval()

        with torch.no_grad():
            samples = self.sampler(n_samples=n_samples)
            samples = samples.squeeze().cpu()
            samples = filter_outliers(samples)

        # Simple text-based evaluation
        mean = samples.mean().item()
        std = samples.std().item()
        print(f"  Samples: mean={mean:.4f}, std={std:.4f}")

        if self.ground_truth_trajectory is not None:
            gt = self.ground_truth_trajectory.squeeze()
            gt_mean = gt.mean().item()
            gt_std = gt.std().item()
            print(f"  Ground truth: mean={gt_mean:.4f}, std={gt_std:.4f}")

    def save(self) -> None:
        """Save model checkpoint."""
        save_checkpoint(
            self.output_dir / "model.pt",
            self.model,
            self.optimizer,
            self.current_epoch,
            extra_data={"loss_history": self.loss_history, "config": self.config},
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        data_loader,
        output_dir: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        ground_truth_trajectory: Optional[torch.Tensor] = None,
    ) -> "DDPMTrainer":
        """
        Create a trainer from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        data_loader : DataLoader
            Training data loader.
        output_dir : str or Path
            Output directory.
        device : str or torch.device
            Device for training.
        ground_truth_trajectory : torch.Tensor, optional
            Ground truth for evaluation.

        Returns
        -------
        DDPMTrainer
            Configured trainer instance.
        """
        model_config = config.get("model_kwargs", {})
        model = ScoreBasedDDPM.from_config(model_config)

        return cls(
            model=model,
            data_loader=data_loader,
            config=config,
            output_dir=output_dir,
            device=device,
            ground_truth_trajectory=ground_truth_trajectory,
        )

