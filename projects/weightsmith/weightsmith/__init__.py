"""weightsmith — post-training weight quantization from scratch.

Shrink a model 2× with provable quality: symmetric int8 schemes
(per-tensor, per-row, per-channel) plus an fp16 baseline, quality metrics
(snr, argmax agreement), a calibrator that picks the smallest mode clearing
your floor, and a whole-state pipeline with byte accounting. Pure python
standard library — lists and floats, no numpy required.
"""
from .calibrate import CalibrationResult, calibrate_tensor, measure_mode
from .errors import MetricsError, QuantError, WeightsmithError
from .metrics import argmax_agreement, max_abs_error, mean_abs_error, snr_db
from .pipeline import Report, TensorReport, quantize_state_dict
from .quant import MODES, QuantizedTensor, quantize_tensor

__all__ = [
    "CalibrationResult", "MODES", "MetricsError", "QuantError", "QuantizedTensor",
    "Report", "TensorReport", "WeightsmithError", "argmax_agreement", "calibrate_tensor",
    "max_abs_error", "mean_abs_error", "measure_mode", "quantize_state_dict",
    "quantize_tensor", "snr_db",
]
__version__ = "1.0.0"
