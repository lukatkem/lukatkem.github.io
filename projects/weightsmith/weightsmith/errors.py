"""Shared exceptions for weightsmith."""


class WeightsmithError(RuntimeError):
    """Base class for all weightsmith errors."""


class QuantError(WeightsmithError):
    """Raised for invalid tensors, unknown modes, or bad calibration floors."""


class MetricsError(WeightsmithError):
    """Raised for empty, ragged, non-finite, or shape-mismatched metric inputs."""
