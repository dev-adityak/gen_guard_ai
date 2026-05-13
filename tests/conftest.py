"""
Pytest configuration and shared fixtures for GenGuard-AI tests.
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from genGuard.pipeline import GenGuardPipeline, PipelineConfig
from genGuard.model_layer.watermarker import GumbelWatermarker, WatermarkConfig
from genGuard.output_layer.pii_detector import PIIDetector


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        "no_pii": "The weather is nice today.",
        "with_email": "Contact me at john@example.com",
        "with_phone": "Call me at 555-123-4567",
        "with_ssn": "SSN: 123-45-6789",
        "with_name": "My name is John Doe",
        "multiple_pii": "John Doe (john@example.com, 555-123-4567) SSN: 123-45-6789",
        "medical": "Patient John Doe has diabetes and takes metformin.",
        "legal": "Attorney Jane Smith filed case number 12345."
    }


@pytest.fixture
def watermark_config():
    """Watermark configuration for testing."""
    return WatermarkConfig(green_fraction=0.25)


@pytest.fixture
def watermarker(watermark_config):
    """Create a watermarker instance for testing."""
    wm = GumbelWatermarker(config=watermark_config)
    wm.set_key(42)
    return wm


@pytest.fixture
def pipeline_config():
    """Pipeline configuration for testing."""
    return PipelineConfig(
        enable_watermarking=True,
        enable_pii_detection=True,
        privacy_budget=8.0,
        watermark_key=42,
        load_model=False
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create a pipeline instance for testing."""
    return GenGuardPipeline(config=pipeline_config)


@pytest.fixture
def pii_detector():
    """Create a PII detector instance for testing."""
    return PIIDetector()