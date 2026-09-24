"""The quantization core — an fp16 baseline and three symmetric int8 schemes.

A tensor goes in as nested python lists of floats and comes back as a
QuantizedTensor: an integer payload plus one or more scales. Dequantization
is a single multiply, so every int8 scheme's worst-case error is exactly
half its scale — a bound the tests check directly.

Scales are symmetric: scale = absmax / 127, so the payload always fits in
[-127, 127] and zero maps to zero. A 1D tensor is treated as a single row,
so int8-per-row collapses to one scale and int8-per-channel to one scale
per element. Everything is pure python — lists, ints, floats — no numpy
required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Union

from .errors import QuantError

MODES: Tuple[str, ...] = ("fp16", "int8-per-tensor", "int8-per-row", "int8-per-channel")
INT8_MODES: Tuple[str, ...] = ("int8-per-tensor", "int8-per-row", "int8-per-channel")

_QMAX = 127
_SCALE_BYTES = 2  # scales are accounted as f16 on disk


def _round_half_away(x: float) -> int:
    """Round to nearest, ties away from zero — deterministic and symmetric."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def _check_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuantError(f"{where} is not a number: {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise QuantError(f"{where} is not finite: {value!r}")
    return v


def _as_grid(values: object) -> Tuple[List[List[float]], Tuple[int, ...]]:
    """Validate a 1D or 2D tensor and normalize it to a grid of float rows.

    A 1D tensor becomes a single row. Ragged rows, empty tensors, mixed
    nesting (3D+), non-numeric, and non-finite values all raise QuantError.
    """
    if not isinstance(values, (list, tuple)):
        raise QuantError(f"tensor must be a list, got {type(values).__name__}")
    if len(values) == 0:
        raise QuantError("empty tensor: need at least one value")
    if isinstance(values[0], (list, tuple)):
        width = None
        grid: List[List[float]] = []
        for r, row in enumerate(values):
            if not isinstance(row, (list, tuple)):
                raise QuantError(f"ragged tensor: row 0 is a list but row {r} is {type(row).__name__}")
            if len(row) == 0:
                raise QuantError(f"empty tensor: row {r} has no values")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise QuantError(f"ragged tensor: row {r} has {len(row)} values, expected {width}")
            grid.append([_check_number(v, f"row {r} col {c}") for c, v in enumerate(row)])
        return grid, (len(grid), width)
    return [[_check_number(v, f"col {c}") for c, v in enumerate(values)]], (len(values),)


def _scale_for(group: List[float]) -> float:
    """Symmetric scale for one group of values: absmax/127 (1.0 if all zero)."""
    absmax = 0.0
    for v in group:
        a = -v if v < 0.0 else v
        if a > absmax:
            absmax = a
    return absmax / _QMAX if absmax > 0.0 else 1.0


def _quantize_group(group: List[float], scale: float) -> List[int]:
    return [max(-_QMAX, min(_QMAX, _round_half_away(v / scale))) for v in group]


@dataclass
class QuantizedTensor:
    """A quantized tensor: the payload plus the scales that undo it.

    data holds the payload — integers in [-127, 127] for the int8 modes;
    the original floats, stored as-is, for the fp16 baseline (marked exact,
    so tests can use it as a round-trip reference). scales holds one scale
    per tensor / row / column depending on mode. `dtype` tags which kind of
    payload is stored, and `size_bytes` is the on-disk accounting.
    """

    shape: Tuple[int, ...]
    mode: str
    data: List[float]
    scales: List[float]

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise QuantError(f"unknown mode {self.mode!r} — expected one of: {', '.join(MODES)}")

    @property
    def dtype(self) -> str:
        """Payload tag: "int8" for the compressed modes, "fp16" for the baseline."""
        return "fp16" if self.mode == "fp16" else "int8"

    @property
    def size_bytes(self) -> int:
        """On-disk accounting: 1 byte per int8 value, 2 bytes per f16 scale.

        The fp16 baseline stores each original value at 2 bytes — it is the
        uncompressed reference the int8 modes are measured against.
        """
        if self.mode == "fp16":
            return 2 * len(self.data)
        return len(self.data) + _SCALE_BYTES * len(self.scales)

    def dequantize(self) -> Union[List[float], List[List[float]]]:
        """Undo the quantization: 2D tensors come back as rows, 1D as a flat list."""
        rows, cols = (1, self.shape[0]) if len(self.shape) == 1 else self.shape
        out: List[List[float]] = []
        for r in range(rows):
            chunk = self.data[r * cols:(r + 1) * cols]
            if self.mode == "fp16":
                out.append(list(chunk))
            elif self.mode == "int8-per-tensor":
                s = self.scales[0]
                out.append([q * s for q in chunk])
            elif self.mode == "int8-per-row":
                s = self.scales[r]
                out.append([q * s for q in chunk])
            else:  # int8-per-channel: the scale index is the column
                out.append([q * self.scales[c] for c, q in enumerate(chunk)])
        return out[0] if len(self.shape) == 1 else out


def quantize_tensor(values: Union[List[float], List[List[float]]], mode: str) -> QuantizedTensor:
    """Quantize a 1D or 2D tensor with one of the MODES.

    "fp16" stores the values as-is (exact baseline); "int8-per-tensor" keeps
    one scale for the whole tensor; "int8-per-row" one per row; and
    "int8-per-channel" one per column. Ragged rows, empty tensors, non-finite
    values, and unknown modes raise QuantError.
    """
    if mode not in MODES:
        raise QuantError(f"unknown mode {mode!r} — expected one of: {', '.join(MODES)}")
    grid, shape = _as_grid(values)
    if mode == "fp16":
        return QuantizedTensor(shape=shape, mode=mode,
                               data=[v for row in grid for v in row], scales=[])
    if mode == "int8-per-tensor":
        scale = _scale_for([v for row in grid for v in row])
        data = [q for row in grid for q in _quantize_group(row, scale)]
        return QuantizedTensor(shape=shape, mode=mode, data=data, scales=[scale])
    if mode == "int8-per-row":
        scales = [_scale_for(row) for row in grid]
        data = [q for row, s in zip(grid, scales) for q in _quantize_group(row, s)]
        return QuantizedTensor(shape=shape, mode=mode, data=data, scales=scales)
    # int8-per-channel: one scale per column of the grid
    columns = [[row[c] for row in grid] for c in range(shape[-1])]
    scales = [_scale_for(column) for column in columns]
    data = [max(-_QMAX, min(_QMAX, _round_half_away(grid[r][c] / scales[c])))
            for r in range(len(grid)) for c in range(shape[-1])]
    return QuantizedTensor(shape=shape, mode=mode, data=data, scales=scales)
