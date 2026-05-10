"""
Utility Module
Common helper functions for GenGuard-AI.
"""

import torch
import logging
from typing import Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_memory_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def validate_config(config: Any) -> bool:
    """Validate configuration object."""
    required_attrs = ['epsilon', 'delta', 'max_grad_norm']
    return all(hasattr(config, attr) for attr in required_attrs)


class DeviceManager:
    """Manages device selection for model training/inference."""

    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @staticmethod
    def get_device_name(device: torch.device) -> str:
        """Get human-readable device name."""
        if device.type == "cuda":
            return f"GPU {torch.cuda.get_device_name(device.index or 0)}"
        elif device.type == "mps":
            return "Apple Silicon GPU"
        return "CPU"

    @staticmethod
    def clear_memory() -> None:
        """Clear GPU memory if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()