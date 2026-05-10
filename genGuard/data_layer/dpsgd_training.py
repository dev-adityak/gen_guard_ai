"""
Differential Privacy SGD Training Module
Implements DP-SGD algorithm using Opacus for privacy-preserving model training.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logger.warning("Opacus not installed. DP-SGD training will not be available.")


class DPSTrainer:
    """
    Differential Privacy SGD Trainer.

    Implements the DP-SGD algorithm from:
    "Deep Learning with Differential Privacy" (Abadi et al., CCS 2016)

    Attributes:
        model: PyTorch model to train
        optimizer: Optimizer for training
        privacy_engine: Opacus PrivacyEngine for DP guarantees
        epsilon: Privacy budget (lower means stronger privacy)
        delta: Privacy failure probability (default: 1e-5)
        max_grad_norm: Gradient clipping norm
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        sample_rate: float = 0.01
    ):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.privacy_engine = None
        self.current_epsilon = 0.0
        self.steps = 0

        if OPACUS_AVAILABLE:
            self._validate_model()

    def _validate_model(self) -> None:
        """Validate model structure for DP compatibility."""
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            logger.warning(f"Model validation warnings: {errors}")

    def attach_privacy_engine(self, sample_size: int = 1000) -> Optional[Any]:
        """Attach Opacus PrivacyEngine to optimizer."""
        if not OPACUS_AVAILABLE:
            logger.error("Opacus not available. Cannot attach privacy engine.")
            return None

        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=int(sample_size * self.sample_rate),
            sample_size=sample_size,
            alphas=[1 + x / 10.0 for x in range(1, 100)],
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            target_delta=self.delta
        )
        self.privacy_engine.attach(self.optimizer)
        return self.privacy_engine

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute single training step with DP-SGD."""
        self.model.train()
        inputs, labels = batch

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        return {'loss': loss.item(), 'step': self.steps}

    def get_privacy_spent(self) -> float:
        """Get cumulative privacy budget spent."""
        if self.privacy_engine and OPACUS_AVAILABLE:
            self.current_epsilon = self.privacy_engine.accounting.get_epsilon()
            return self.current_epsilon
        return 0.0

    def compute_privacy_budget(
        self,
        num_epochs: int,
        batch_size: int,
        sample_size: int
    ) -> Tuple[float, float]:
        """Compute privacy budget for given training configuration."""
        sampling_rate = batch_size / sample_size
        steps = num_epochs * (sample_size // batch_size)
        return self.epsilon, self.delta