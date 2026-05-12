"""
GenGuard-AI Unified Pipeline
Combines all layers into a single processing pipeline.
"""

import logging
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .data_layer.dpsgd_training import DPSTrainer
from .data_layer.privacy_tracker import PrivacyBudgetTracker
from .data_layer.fl_simulated import FederatedTrainer

from .output_layer.pii_detector import PIIDetector, PIIAnalysisResult
from .output_layer.context_filter import ContextAwareFilter
from .output_layer.redaction_engine import RedactionEngine, RedactionResult

from .model_layer.watermarker import GumbelWatermarker, WatermarkConfig
from .model_layer.model_quantizer import ModelQuantizer

from .monitoring.attack_simulator import AttackSimulator
from .monitoring.audit_logger import AuditLogger
from .monitoring.privacy_health import PrivacyHealthMonitor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the unified pipeline."""
    enable_dp_training: bool = True
    enable_watermarking: bool = True
    enable_pii_detection: bool = True
    enable_monitoring: bool = True
    privacy_budget: float = 8.0
    watermark_key: Optional[int] = None


class GenGuardPipeline:
    """
    Unified GenGuard-AI Pipeline.

    Combines:
    - Data Layer (DP-SGD, FL, Privacy Tracking)
    - Output Layer (PII Detection, Redaction)
    - Model Layer (Watermarking, Quantization)
    - Monitoring (Attack Simulation, Audit Logging)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model=None,
        tokenizer=None
    ):
        self.config = config or PipelineConfig()
        self.model = model
        self.tokenizer = tokenizer

        self._init_components()
        logger.info("GenGuard-AI Pipeline initialized")

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        self.privacy_tracker = PrivacyBudgetTracker(
            target_epsilon=self.config.privacy_budget
        )

        if self.model is not None:
            self.dpsgd_trainer = DPSTrainer(
                model=self.model,
                optimizer=None,
                epsilon=self.config.privacy_budget
            )
        else:
            self.dpsgd_trainer = None

        self.fl_trainer = None
        self.pii_detector = PIIDetector()
        self.context_filter = ContextAwareFilter()
        self.redaction_engine = RedactionEngine(
            pii_detector=self.pii_detector,
            context_filter=self.context_filter
        )

        if self.config.enable_watermarking:
            watermark_config = WatermarkConfig()
            self.watermarker = GumbelWatermarker(config=watermark_config)
            if self.config.watermark_key:
                self.watermarker.set_key(self.config.watermark_key)
        else:
            self.watermarker = None

        self.quantizer = ModelQuantizer()
        self.attack_simulator = AttackSimulator(
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.audit_logger = AuditLogger()
        self.health_monitor = PrivacyHealthMonitor(
            target_epsilon=self.config.privacy_budget
        )

    def process_output(
        self,
        text: str,
        context: str = "",
        watermark: bool = True
    ) -> Dict[str, Any]:
        """
        Process LLM output through the pipeline.

        Args:
            text: Generated text to process
            context: Context hint for filtering
            watermark: Whether to embed watermark

        Returns:
            Dictionary with processed output and metadata
        """
        result = self.redaction_engine.process(text, context)

        if watermark and self.watermarker is not None:
            result.redacted_text = self._embed_watermark_text(
                result.redacted_text,
                context
            )

        self.audit_logger.log_redaction(
            original_text=text,
            redacted_text=result.redacted_text,
            entities=result.entities_redacted,
            context=context
        )

        return {
            'original': text,
            'output': result.redacted_text,
            'pii_found': result.entities_found,
            'pii_redacted': result.entities_redacted,
            'explanation': result.explanation,
            'context': context,
            'confidence': result.confidence
        }

    def _embed_watermark_text(
        self,
        text: str,
        context: str
    ) -> str:
        """Embed watermark into text (simplified)."""
        if self.watermarker is None or self.tokenizer is None:
            return text

        tokens = self.tokenizer.encode(text)
        watermarked_logits = torch.randn(len(tokens), self.tokenizer.vocab_size)

        watermarked_logits = self.watermarker.embed_watermark(watermarked_logits.unsqueeze(0))
        watermarked_tokens = watermarked_logits.argmax(dim=-1).squeeze()[:len(tokens)]

        return self.tokenizer.decode(watermarked_tokens.tolist())

    def run_privacy_evaluation(self) -> Dict[str, Any]:
        """Run privacy evaluation (attack simulation)."""
        health_metrics = self.attack_simulator.compute_privacy_health_score()

        self.health_monitor.update_metrics(
            epsilon=self.privacy_tracker.get_current_epsilon(),
            jailbreak_rate=health_metrics['jailbreak_rate'],
            mia_score=health_metrics['mia_score']
        )

        return health_metrics

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of all pipeline components."""
        return {
            'privacy': self.privacy_tracker.get_summary(),
            'health': self.health_monitor.get_summary(),
            'attacks_tested': len(self.attack_simulator.attack_history),
            'audit_events': len(self.audit_logger.events)
        }

    def export_reports(self, output_dir: str = "reports") -> None:
        """Export all reports."""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self.privacy_tracker.export_json(output_path / "privacy_report.json")
        self.audit_logger.export_json(output_path / "audit_report.json")
        self.health_monitor.export_snapshots(output_path / "health_report.json")
        self.attack_simulator.export_attack_history(str(output_path / "attack_history.json"))

        logger.info(f"Reports exported to {output_path}")