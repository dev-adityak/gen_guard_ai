"""
Audit Logger Module
Records all redaction events and compliance data for audit reporting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Single audit event record."""
    timestamp: str
    event_type: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AuditLogger:
    """
    Audit Logger for Compliance Reporting.

    Records:
    - Redaction events
    - Model access logs
    - Privacy budget updates
    - Policy violations

    Supports export in multiple formats.
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        rotation: str = "daily"
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rotation = rotation

        self.events: List[AuditEvent] = []
        self.event_counts = defaultdict(int)

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up file logging."""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger.addHandler(handler)
        self.logger = logger

    def log_redaction(
        self,
        original_text: str,
        redacted_text: str,
        entities: List[str],
        context: str = ""
    ) -> None:
        """Log a redaction event."""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type="redaction",
            details={
                'original_length': len(original_text),
                'redacted_length': len(redacted_text),
                'entities_redacted': entities,
                'context': context,
                'original_preview': original_text[:200]
            }
        )
        self._add_event(event)

    def log_model_access(
        self,
        model_id: str,
        operation: str,
        user_id: Optional[str] = None
    ) -> None:
        """Log model access event."""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type="model_access",
            details={
                'model_id': model_id,
                'operation': operation
            },
            user_id=user_id
        )
        self._add_event(event)

    def log_privacy_budget_update(
        self,
        epsilon: float,
        delta: float,
        step: int
    ) -> None:
        """Log privacy budget update."""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type="privacy_budget",
            details={
                'epsilon': epsilon,
                'delta': delta,
                'step': step
            }
        )
        self._add_event(event)

    def log_policy_violation(
        self,
        violation_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Log policy violation."""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type="policy_violation",
            details={
                'violation_type': violation_type,
                **details
            }
        )
        self._add_event(event)
        self.logger.warning(f"Policy violation detected: {violation_type}")

    def _add_event(self, event: AuditEvent) -> None:
        """Add event to audit log."""
        self.events.append(event)
        self.event_counts[event.event_type] += 1

        self.logger.info(f"Audit event: {event.event_type}")

    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of all logged events."""
        return {
            'total_events': len(self.events),
            'event_counts': dict(self.event_counts),
            'first_event': self.events[0].timestamp if self.events else None,
            'last_event': self.events[-1].timestamp if self.events else None,
            'log_dir': str(self.log_dir)
        }

    def export_json(self, filepath: Optional[Path] = None) -> Path:
        """Export audit log to JSON."""
        if filepath is None:
            filepath = self.log_dir / f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_event_summary(),
                'events': [asdict(e) for e in self.events]
            }, f, indent=2)

        return filepath

    def export_csv(self, filepath: Optional[Path] = None) -> Path:
        """Export audit log to CSV."""
        if filepath is None:
            filepath = self.log_dir / f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        lines = ["timestamp,event_type,details"]
        for event in self.events:
            details_str = json.dumps(event.details).replace(',', ';')
            lines.append(f"{event.timestamp},{event.event_type},{details_str}")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        return filepath

    def generate_compliance_report(
        self,
        regulations: List[str] = ["GDPR", "HIPAA"]
    ) -> Dict[str, Any]:
        """
        Generate compliance report.

        Args:
            regulations: List of regulations to check compliance for

        Returns:
            Compliance report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'regulations': regulations,
            'summary': self.get_event_summary(),
            'compliance_checks': {}
        }

        redaction_count = self.event_counts.get('redaction', 0)
        policy_violations = self.event_counts.get('policy_violation', 0)

        report['compliance_checks']['gdpr'] = {
            'compliant': policy_violations == 0,
            'redaction_events': redaction_count,
            'violations': policy_violations
        }

        report['compliance_checks']['hipaa'] = {
            'compliant': policy_violations == 0,
            'phi_redactions': redaction_count,
            'violations': policy_violations
        }

        return report