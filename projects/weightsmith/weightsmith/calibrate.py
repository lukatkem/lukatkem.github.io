"""Calibration — try every mode, keep the smallest one that clears the floor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .errors import QuantError
from .metrics import Tensor, argmax_agreement, snr_db
from .quant import MODES, quantize_tensor


@dataclass
class CalibrationResult:
    """One mode's scorecard against the original tensor."""

    mode: str
    snr_db: float
    agreement: float
    size_bytes: int


def measure_mode(values: Tensor, mode: str) -> CalibrationResult:
    """Quantize with one mode and score it: snr, argmax agreement, on-disk size."""
    quantized = quantize_tensor(values, mode)
    restored = quantized.dequantize()
    return CalibrationResult(
        mode=mode,
        snr_db=snr_db(values, restored),
        agreement=argmax_agreement(values, restored),
        size_bytes=quantized.size_bytes,
    )


def calibrate_tensor(values: Tensor, min_agreement: float = 0.95) -> CalibrationResult:
    """Pick the smallest mode whose argmax agreement clears `min_agreement`.

    Every mode is measured; among those with agreement >= min_agreement the
    one with the fewest bytes wins (ties go to the better agreement, then to
    the canonical mode order). The fp16 baseline is exact, so any floor in
    [0, 1] always has at least one candidate — a tight floor degrades
    gracefully to fp16 instead of failing.
    """
    if isinstance(min_agreement, bool) or not isinstance(min_agreement, (int, float)) \
            or not 0.0 <= float(min_agreement) <= 1.0:
        raise QuantError(f"min_agreement must be a number in [0, 1], got {min_agreement!r}")
    best: Optional[CalibrationResult] = None
    for mode in MODES:
        result = measure_mode(values, mode)
        if result.agreement < min_agreement:
            continue
        if best is None or (result.size_bytes, -result.agreement) < (best.size_bytes, -best.agreement):
            best = result
    if best is None:  # unreachable: fp16 is exact and always qualifies
        raise QuantError("no mode meets the quality floor")
    return best
