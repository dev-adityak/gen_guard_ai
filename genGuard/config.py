"""
GenGuard-AI Configuration Module
Centralized configuration for all framework components.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class PrivacyConfig:
    """Differential Privacy configuration."""
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_epsilon: float = 8.0
    target_delta: float = 1e-5


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5
    fraction_fit: float = 1.0
    aggregation_method: str = "fedavg"
    partition_type: str = "iid"


@dataclass
class PIIConfig:
    """PII Detection configuration."""
    language: str = "en"
    nlp_model: str = "en_core_web_sm"
    confidence_threshold: float = 0.5
    supported_entities: List[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL", "PHONE_NUMBER", "SSN",
        "CREDIT_CARD", "DATE_TIME", "US_BANK_ROUTING_NUMBER"
    ])


@dataclass
class FilterConfig:
    """Context-aware filter configuration."""
    model_path: Optional[str] = None
    num_labels: int = 2
    threshold: float = 0.5


@dataclass
class WatermarkConfig:
    """Watermarking configuration."""
    watermark_length: int = 50
    gumbel_hard: bool = True
    gumbel_temp: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: Path = LOGS_DIR
    privacy_log_dir: Path = field(default_factory=lambda: LOGS_DIR / "privacy")
    redaction_log_dir: Path = field(default_factory=lambda: LOGS_DIR / "redactions")
    audit_log_path: Path = field(default_factory=lambda: LOGS_DIR / "audit.json")