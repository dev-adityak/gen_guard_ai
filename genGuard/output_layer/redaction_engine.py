"""
Redaction Engine Module
Unified redaction with self-explanation generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class RedactionEvent:
    """Record of a redaction event."""
    timestamp: str
    original_text: str
    redacted_text: str
    entities_redacted: List[str]
    reason: str
    context: str


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    original_text: str
    redacted_text: str
    entities_found: List[str]
    entities_redacted: List[str]
    explanation: str
    context: str
    confidence: float


class RedactionEngine:
    """
    Unified Redaction Engine.

    Combines:
    - PII detection
    - Context-aware filtering
    - Self-explanation generation
    - Audit logging
    """

    def __init__(
        self,
        pii_detector=None,
        context_filter=None,
        log_path: str = "logs/redactions"
    ):
        from .pii_detector import PIIDetector
        from .context_filter import ContextAwareFilter

        self.pii_detector = pii_detector or PIIDetector()
        self.context_filter = context_filter or ContextAwareFilter()

        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.audit_log: List[RedactionEvent] = []
        self.logger = logging.getLogger(__name__)

    def process(
        self,
        text: str,
        context: str = "",
        auto_redact: bool = True
    ) -> RedactionResult:
        """Process text through redaction pipeline."""
        pii_result = self.pii_detector.detect(text)

        entities_found = [e.entity_type for e in pii_result.entities]

        if not pii_result.has_pii:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                entities_found=[],
                entities_redacted=[],
                explanation="No PII detected",
                context=context,
                confidence=1.0
            )

        entities_to_redact = []
        for entity in pii_result.entities:
            should_redact, reason = self.context_filter.should_redact(
                text,
                pii_result.entities,
                context
            )
            if should_redact:
                entities_to_redact.append(entity)

        entities_redacted = [e.entity_type for e in entities_to_redact]

        redacted_text = text
        if auto_redact and entities_to_redact:
            redacted_text = self.pii_detector.redact(text)

        explanation = self._generate_explanation(
            entities_redacted,
            context,
            pii_result.risk_level
        )

        event = RedactionEvent(
            timestamp=datetime.now().isoformat(),
            original_text=text[:500],
            redacted_text=redacted_text[:500],
            entities_redacted=entities_redacted,
            reason=explanation,
            context=context
        )
        self._log_event(event)

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            entities_found=entities_found,
            entities_redacted=entities_redacted,
            explanation=explanation,
            context=context,
            confidence=pii_result.risk_level != "none"
        )

    def _generate_explanation(
        self,
        entities_redacted: List[str],
        context: str,
        risk_level: str
    ) -> str:
        """Generate human-readable explanation."""
        if not entities_redacted:
            return "No redaction needed - no PII detected"

        entity_summary = ", ".join(set(entities_redacted))

        if risk_level == "high":
            return f"Redacted {entity_summary} due to high-risk data (GDPR/HIPAA compliance)"
        elif risk_level == "medium":
            return f"Redacted {entity_summary} for privacy protection (GDPR article 5)"
        else:
            return f"Redacted {entity_summary} for data minimization"

    def _log_event(self, event: RedactionEvent) -> None:
        """Log redaction event to audit trail."""
        self.audit_log.append(event)

        log_file = self.log_path / f"redaction_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def export_audit_log(self, filepath: Optional[Path] = None) -> Path:
        """Export complete audit log."""
        if filepath is None:
            filepath = self.log_path / f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, 'w') as f:
            json.dump([asdict(e) for e in self.audit_log], f, indent=2)

        return filepath