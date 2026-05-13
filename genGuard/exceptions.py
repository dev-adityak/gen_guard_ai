"""
GenGuard-AI Custom Exceptions
"""

class GenGuardException(Exception):
    """Base exception for all GenGuard-AI errors."""
    pass


class ModelNotLoadedError(GenGuardException):
    """Raised when model is not loaded but required."""
    pass


class PIIDetectionError(GenGuardException):
    """Raised when PII detection fails."""
    pass


class RedactionError(GenGuardException):
    """Raised when text redaction fails."""
    pass


class WatermarkError(GenGuardException):
    """Raised when watermark operations fail."""
    pass


class PrivacyBudgetExceededError(GenGuardException):
    """Raised when privacy budget is exceeded."""
    pass


class ConfigurationError(GenGuardException):
    """Raised when configuration is invalid."""
    pass


class PipelineError(GenGuardException):
    """Raised when pipeline processing fails."""
    pass