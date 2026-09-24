"""Calibration — floor semantics, winner selection, determinism."""
from __future__ import annotations

import pytest

from weightsmith import MODES, QuantError, calibrate_tensor, measure_mode, quantize_tensor
from weightsmith.cli import lcg_floats

WELL_BEHAVED = [lcg_floats(11 + r, 16, -0.6, 0.6) for r in range(16)]


def test_calibrate_picks_fp16_at_tight_floor(highfloor_tensor):
    result = calibrate_tensor(highfloor_tensor, min_agreement=0.9999)
    assert result.mode == "fp16"          # every int8 scheme flips an argmax here
    assert result.agreement == 1.0
    assert result.size_bytes == 2 * 4 * 6


def test_calibrate_picks_per_row_when_relaxed(outlier_row_tensor):
    result = calibrate_tensor(outlier_row_tensor, min_agreement=0.9)
    assert result.mode == "int8-per-row"  # per-tensor (0.375) fails; per-row (1.0) is smallest
    assert result.size_bytes == 8 * 12 + 2 * 8
    assert result.agreement >= 0.9


def test_calibrate_picks_per_channel_when_relaxed(outlier_col_tensor):
    result = calibrate_tensor(outlier_col_tensor, min_agreement=0.9)
    assert result.mode == "int8-per-channel"  # per-tensor (0.5) fails; per-channel is smallest
    assert result.size_bytes == 12 * 8 + 2 * 8
    assert result.agreement >= 0.9


def test_calibrate_picks_an_int8_mode_on_well_behaved_tensors():
    result = calibrate_tensor(WELL_BEHAVED, min_agreement=0.95)
    assert result.mode.startswith("int8")          # fp16 is never the answer here
    assert result.size_bytes < quantize_tensor(WELL_BEHAVED, "fp16").size_bytes
    assert result.agreement >= 0.95


def test_calibrate_default_floor_is_095(outlier_row_tensor):
    assert calibrate_tensor(outlier_row_tensor) == calibrate_tensor(outlier_row_tensor, 0.95)


def test_calibrate_result_matches_measure_mode(highfloor_tensor):
    result = calibrate_tensor(highfloor_tensor, min_agreement=0.9999)
    direct = measure_mode(highfloor_tensor, result.mode)
    assert (result.mode, result.snr_db, result.agreement, result.size_bytes) == \
        (direct.mode, direct.snr_db, direct.agreement, direct.size_bytes)
    assert result.mode in MODES


def test_calibrate_is_deterministic(outlier_row_tensor):
    a = calibrate_tensor(outlier_row_tensor, 0.9)
    b = calibrate_tensor(outlier_row_tensor, 0.9)
    assert a == b


def test_calibrate_invalid_floor_raises(highfloor_tensor):
    for bad in (1.5, -0.1):
        with pytest.raises(QuantError, match="min_agreement"):
            calibrate_tensor(highfloor_tensor, bad)
