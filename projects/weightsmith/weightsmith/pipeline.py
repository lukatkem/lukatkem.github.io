"""Whole-model pipeline — a state dict in, quantized tensors and a report out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .calibrate import CalibrationResult, calibrate_tensor, measure_mode
from .errors import QuantError
from .metrics import Tensor
from .quant import MODES, QuantizedTensor, quantize_tensor

_FP16_BYTES = 2  # the baseline the report accounts originals at


@dataclass
class TensorReport:
    """Per-tensor line of the report: quality, mode, and byte accounting."""

    name: str
    mode: str
    snr_db: float
    agreement: float
    original_bytes: int
    quantized_bytes: int

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.quantized_bytes if self.quantized_bytes else 0.0


@dataclass
class Report:
    """Byte totals over a quantized state dict.

    original_bytes counts each element at the fp16 baseline (2 bytes) — the
    same accounting the fp16 mode uses on disk — so `ratio` reads as "how
    much smaller than the fp16 checkpoint the quantized state ships as".
    """

    tensors: List[TensorReport]

    @property
    def original_bytes(self) -> int:
        return sum(t.original_bytes for t in self.tensors)

    @property
    def quantized_bytes(self) -> int:
        return sum(t.quantized_bytes for t in self.tensors)

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.quantized_bytes if self.quantized_bytes else 0.0


def _elements(shape: Tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def quantize_state_dict(state: Dict[str, Tensor], mode: Optional[str] = None,
                        min_agreement: float = 0.95) -> Tuple[Dict[str, QuantizedTensor], Report]:
    """Quantize every tensor in a state dict.

    With `mode` set, every tensor uses that mode. With `mode=None` each
    tensor is calibrated independently against `min_agreement`. Returns
    (quantized, report): a dict of QuantizedTensor plus a Report with
    per-tensor quality lines and byte totals.
    """
    if not isinstance(state, dict) or not state:
        raise QuantError("state must be a non-empty dict of {name: tensor}")
    if mode is not None and mode not in MODES:
        raise QuantError(f"unknown mode {mode!r} — expected one of: {', '.join(MODES)}")
    quantized: Dict[str, QuantizedTensor] = {}
    entries: List[TensorReport] = []
    for name, values in state.items():
        if mode is None:
            score: CalibrationResult = calibrate_tensor(values, min_agreement)
            qt = quantize_tensor(values, score.mode)
        else:
            qt = quantize_tensor(values, mode)
            score = measure_mode(values, mode)
        quantized[name] = qt
        entries.append(TensorReport(
            name=name, mode=qt.mode, snr_db=score.snr_db, agreement=score.agreement,
            original_bytes=_elements(qt.shape) * _FP16_BYTES,
            quantized_bytes=qt.size_bytes))
    return quantized, Report(tensors=entries)
