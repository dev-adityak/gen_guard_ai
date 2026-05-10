"""
Privacy Health Monitor Module
Computes and tracks privacy health metrics over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Snapshot of privacy health at a point in time."""
    timestamp: str
    epsilon: float
    jailbreak_rate: float
    mia_score: float
    health_score: float
    events: int


class PrivacyHealthMonitor:
    """
    Privacy Health Monitor.

    Tracks:
    - Privacy budget consumption (epsilon)
    - Attack success rates
    - Membership inference scores
    - Overall health score

    Generates periodic health snapshots.
    """

    def __init__(
        self,
        target_epsilon: float = 8.0,
        log_dir: str = "logs/health"
    ):
        self.target_epsilon = target_epsilon
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: List[HealthSnapshot] = []
        self.logger = logging.getLogger(__name__)

        self.current_epsilon = 0.0
        self.jailbreak_rate = 0.0
        self.mia_score = 0.0

    def update_metrics(
        self,
        epsilon: Optional[float] = None,
        jailbreak_rate: Optional[float] = None,
        mia_score: Optional[float] = None
    ) -> None:
        """Update current metrics."""
        if epsilon is not None:
            self.current_epsilon = epsilon
        if jailbreak_rate is not None:
            self.jailbreak_rate = jailbreak_rate
        if mia_score is not None:
            self.mia_score = mia_score

    def compute_health_score(self) -> float:
        """Compute overall health score (0-100)."""
        epsilon_score = max(0, (1 - self.current_epsilon / self.target_epsilon) * 50)
        attack_score = max(0, (1 - self.jailbreak_rate) * 30)
        mia_penalty = max(0, (1 - self.mia_score) * 20)

        return epsilon_score + attack_score + mia_penalty

    def take_snapshot(self, events_count: int = 0) -> HealthSnapshot:
        """Take a health snapshot."""
        health_score = self.compute_health_score()

        snapshot = HealthSnapshot(
            timestamp=datetime.now().isoformat(),
            epsilon=self.current_epsilon,
            jailbreak_rate=self.jailbreak_rate,
            mia_score=self.mia_score,
            health_score=health_score,
            events=events_count
        )

        self.snapshots.append(snapshot)
        self.logger.info(f"Health snapshot taken: {health_score:.1f}")

        return snapshot

    def get_trend(self) -> Dict[str, Any]:
        """Get health trend over time."""
        if len(self.snapshots) < 2:
            return {'trend': 'insufficient_data'}

        recent = self.snapshots[-5:]
        avg_health = sum(s.health_score for s in recent) / len(recent)
        first_health = self.snapshots[0].health_score
        last_health = self.snapshots[-1].health_score

        if last_health > first_health:
            direction = "improving"
        elif last_health < first_health:
            direction = "declining"
        else:
            direction = "stable"

        return {
            'direction': direction,
            'avg_recent_health': avg_health,
            'first_snapshot': first_health,
            'latest_snapshot': last_health,
            'change': last_health - first_health
        }

    def export_snapshots(self, filepath: Optional[Path] = None) -> Path:
        """Export health snapshots to file."""
        if filepath is None:
            filepath = self.log_dir / f"health_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'target_epsilon': self.target_epsilon,
            'snapshots': [asdict(s) for s in self.snapshots],
            'trend': self.get_trend()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            'current_epsilon': self.current_epsilon,
            'target_epsilon': self.target_epsilon,
            'budget_remaining_pct': max(0, (1 - self.current_epsilon / self.target_epsilon) * 100),
            'jailbreak_rate': self.jailbreak_rate,
            'mia_score': self.mia_score,
            'health_score': self.compute_health_score(),
            'total_snapshots': len(self.snapshots),
            'trend': self.get_trend()
        }