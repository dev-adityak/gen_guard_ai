"""
Privacy Budget Tracker Module
Tracks cumulative privacy loss during DP-SGD training.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class PrivacySpent:
    """Record of privacy budget spent at a point in time."""
    timestamp: str
    step: int
    epsilon: float
    delta: float
    noise_multiplier: float
    batch_size: int


class PrivacyBudgetTracker:
    """
    Tracks and logs privacy budget consumption during training.

    Provides:
    - Real-time privacy budget monitoring
    - Export to audit formats (JSON, CSV)
    - Privacy budget alerts
    """

    def __init__(
        self,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        log_dir: str = "logs/privacy"
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[PrivacySpent] = []
        self.current_step = 0
        self.logger = logging.getLogger(__name__)

    def update(
        self,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        batch_size: int
    ) -> None:
        """Update privacy budget tracking."""
        self.current_step += 1
        entry = PrivacySpent(
            timestamp=datetime.now().isoformat(),
            step=self.current_step,
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=noise_multiplier,
            batch_size=batch_size
        )
        self.history.append(entry)

        self.logger.info(
            f"Privacy budget: epsilon={epsilon:.2f}, delta={delta:.2e}, step={self.current_step}"
        )

        if epsilon >= self.target_epsilon:
            self.logger.warning(f"Target privacy budget reached! epsilon={epsilon:.2f}")

    def get_current_epsilon(self) -> float:
        """Get current privacy budget."""
        if self.history:
            return self.history[-1].epsilon
        return 0.0

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.target_epsilon - self.get_current_epsilon()

    def export_json(self, filepath: Optional[Path] = None) -> Path:
        """Export privacy history to JSON."""
        if filepath is None:
            filepath = self.log_dir / f"privacy_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta,
            'history': [asdict(entry) for entry in self.history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def export_csv(self, filepath: Optional[Path] = None) -> Path:
        """Export privacy history to CSV."""
        if filepath is None:
            filepath = self.log_dir / f"privacy_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        lines = ["timestamp,step,epsilon,delta,noise_multiplier,batch_size"]
        for entry in self.history:
            lines.append(
                f"{entry.timestamp},{entry.step},{entry.epsilon},"
                f"{entry.delta},{entry.noise_multiplier},{entry.batch_size}"
            )

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        return filepath

    def get_summary(self) -> Dict:
        """Get privacy budget summary."""
        final_epsilon = self.get_current_epsilon()
        return {
            'target_epsilon': self.target_epsilon,
            'achieved_epsilon': final_epsilon,
            'target_delta': self.target_delta,
            'total_steps': self.current_step,
            'budget_consumed_pct': (final_epsilon / self.target_epsilon) * 100,
            'within_budget': final_epsilon <= self.target_epsilon
        }