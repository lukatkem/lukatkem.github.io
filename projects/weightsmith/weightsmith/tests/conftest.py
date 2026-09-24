"""Shared fixtures: hand-crafted tensors with known quantization behavior.

Every tensor here is a literal — no randomness — so each test's outcome is
locked by construction:

- outlier_row_tensor: 7 identical-structure rows plus one 25× row. A coarse
  per-tensor scale flattens the normal rows (argmax agreement 0.375) while
  per-row scales keep every row's own resolution (agreement 1.0).
- outlier_col_tensor: the same idea rotated — one huge column — so
  per-channel is the scheme that adapts.
- highfloor_tensor: every int8 scheme flips at least one row's argmax
  (per-tensor 0.75, per-row 0.75, per-channel 0.50), so only the exact fp16
  baseline clears a 0.9999 floor.
"""
from __future__ import annotations

import pytest

_BASE_ROW = [0.52, -0.18, 0.55, -0.61, 0.13, -0.37, 0.42, -0.09, 0.25, -0.44, 0.31, 0.07]
_OUTLIER_ROW = [25.0, -12.0, 6.0, -3.0, 1.5, -0.8, 0.4, -0.2, 0.1, 0.05, -0.3, 0.15]


def _rotated(row: list, k: int) -> list:
    return row[k:] + row[:k]


@pytest.fixture
def outlier_row_tensor() -> list:
    """8x12: seven normal rows (rotations of one base row) + one 25x row."""
    rows = [_rotated(_BASE_ROW, i % 12) for i in range(7)]
    rows.append(list(_OUTLIER_ROW))
    return rows


@pytest.fixture
def outlier_col_tensor() -> list:
    """12x8: seven normal columns + one huge outlier column."""
    return [_rotated(_BASE_ROW[:7], r % 7) + [_OUTLIER_ROW[r]] for r in range(12)]


@pytest.fixture
def highfloor_tensor() -> list:
    """4x6: every int8 scheme flips at least one row's argmax."""
    return [
        [40.0, 0.50, 0.53, 0.10, -0.20, 0.05],
        [-35.0, 0.24, 0.22, 0.12, -0.15, 0.03],
        [4.90, 4.945, 4.95, 0.10, -0.20, 0.05],
        [-22.0, 25.0, 0.39, 0.11, -0.18, 0.04],
    ]
