"""Pipeline — state dict in, quantized tensors and byte-accurate report out."""
from __future__ import annotations

import pytest

from weightsmith import QuantError, QuantizedTensor, Report, quantize_state_dict, quantize_tensor

WELL_BEHAVED = [[0.42, -0.18, 0.55, -0.61], [0.13, -0.37, 0.48, -0.09],
                [0.25, -0.52, 0.31, 0.07], [0.44, -0.28, 0.05, -0.15]]


def _state(outlier_row_tensor):
    return {"encoder.weight": outlier_row_tensor, "encoder.bias": WELL_BEHAVED}


def test_pipeline_explicit_mode(outlier_row_tensor):
    quantized, report = quantize_state_dict(_state(outlier_row_tensor), mode="int8-per-row")
    assert set(quantized) == {"encoder.weight", "encoder.bias"}
    assert all(isinstance(qt, QuantizedTensor) for qt in quantized.values())
    assert all(qt.mode == "int8-per-row" for qt in quantized.values())
    names = [t.name for t in report.tensors]
    assert names == ["encoder.weight", "encoder.bias"]          # order preserved
    total_elements = 8 * 12 + 4 * 4
    assert report.original_bytes == 2 * total_elements          # fp16 baseline accounting
    assert report.quantized_bytes == sum(qt.size_bytes for qt in quantized.values())
    assert report.ratio == pytest.approx(report.original_bytes / report.quantized_bytes)
    assert report.ratio > 1.0
    assert report.quantized_bytes < report.original_bytes


def test_pipeline_calibrated_meets_floor(outlier_row_tensor):
    quantized, report = quantize_state_dict(_state(outlier_row_tensor), mode=None)
    for entry in report.tensors:
        assert entry.agreement >= 0.95          # every pick clears the floor
        assert entry.mode in ("int8-per-tensor", "int8-per-row", "int8-per-channel", "fp16")
    assert quantized["encoder.weight"].mode == "int8-per-row"  # outlier tensor picks per-row
    assert report.ratio > 1.0


def test_pipeline_reports_quality_lines(outlier_row_tensor):
    _, report = quantize_state_dict(_state(outlier_row_tensor), mode="fp16")
    for entry in report.tensors:
        assert entry.snr_db == float("inf")      # fp16 is exact
        assert entry.agreement == 1.0
        assert entry.original_bytes == entry.quantized_bytes


def test_pipeline_rejects_bad_input(outlier_row_tensor):
    with pytest.raises(QuantError):
        quantize_state_dict({})
    with pytest.raises(QuantError):
        quantize_state_dict({"w": outlier_row_tensor}, mode="int4")
    with pytest.raises(QuantError):
        quantize_state_dict([outlier_row_tensor])  # not a dict
