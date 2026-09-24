"""Quantization core — round-trips, error bounds, scale structure, size accounting, errors."""
from __future__ import annotations

import pytest

from weightsmith import MODES, QuantError, QuantizedTensor, quantize_tensor
from weightsmith.cli import lcg_floats

GRID_5X7 = [lcg_floats(5 + r, 7, -1.0, 1.0) for r in range(5)]
ONE_D = [0.5, -0.25, 0.1, -0.4, 0.3, 0.2, -0.6, 0.15]


def _scale_at(qt: QuantizedTensor, r: int, c: int) -> float:
    if qt.mode == "int8-per-tensor":
        return qt.scales[0]
    if qt.mode == "int8-per-row":
        return qt.scales[r]
    return qt.scales[c]


# ---------- round-trips ----------
def test_fp16_round_trip_exact_1d_and_2d():
    for values in ([1.5, -2.25, 0.0, 127.5, -0.001],
                   [[0.42, -0.18, 0.55], [-0.61, 0.13, -0.37]]):
        qt = quantize_tensor(values, "fp16")
        assert qt.dequantize() == values  # stored as-is, bit-exact round-trip
        assert qt.dtype == "fp16"
        assert qt.scales == []


# ---------- payload structure ----------
def test_int8_payload_structure():
    values = [[0.42, -0.18, 0.55, -0.61], [0.13, -0.37, 0.48, -0.09], [0.25, -0.52, 0.31, 0.07]]
    expected_scales = {
        "int8-per-tensor": [0.61 / 127],                                    # one global absmax
        "int8-per-row": [0.61 / 127, 0.48 / 127, 0.52 / 127],               # per-row absmax
        "int8-per-channel": [0.42 / 127, 0.52 / 127, 0.55 / 127, 0.61 / 127],  # per-column absmax
    }
    for mode, scales in expected_scales.items():
        qt = quantize_tensor(values, mode)
        assert qt.dtype == "int8"
        assert qt.scales == pytest.approx(scales)
        assert all(float(q).is_integer() for q in qt.data)          # payload is integral
        assert all(-127 <= q <= 127 for q in qt.data)               # and in int8 range


def test_all_zero_tensor_quantizes_without_dividing_by_zero():
    qt = quantize_tensor([[0.0, 0.0], [0.0, 0.0]], "int8-per-row")
    assert qt.scales == [1.0, 1.0]  # arbitrary scale, data all zero
    assert qt.dequantize() == [[0.0, 0.0], [0.0, 0.0]]


# ---------- provable error bound ----------
def test_int8_error_bound_within_half_scale():
    for mode in ("int8-per-tensor", "int8-per-row", "int8-per-channel"):
        qt = quantize_tensor(GRID_5X7, mode)
        restored = qt.dequantize()
        for r, (row, back) in enumerate(zip(GRID_5X7, restored)):
            for c, (v, d) in enumerate(zip(row, back)):
                assert abs(d - v) <= _scale_at(qt, r, c) / 2 + 1e-9


def test_per_row_scales_adapt_to_rows():
    values = [[0.5, -0.4, 0.3, -0.2], [40.0, -30.0, 20.0, -10.0]]
    qt = quantize_tensor(values, "int8-per-row")
    assert qt.scales[1] == pytest.approx(40.0 / 127)
    assert qt.scales[0] == pytest.approx(0.5 / 127)  # normal row keeps fine resolution


# ---------- which scheme wins where ----------
def test_per_row_beats_per_tensor_snr_on_outlier_row(outlier_row_tensor):
    from weightsmith import snr_db
    per_tensor = quantize_tensor(outlier_row_tensor, "int8-per-tensor").dequantize()
    per_row = quantize_tensor(outlier_row_tensor, "int8-per-row").dequantize()
    assert snr_db(outlier_row_tensor, per_row) > snr_db(outlier_row_tensor, per_tensor)


def test_per_channel_beats_per_tensor_snr_on_outlier_column(outlier_col_tensor):
    from weightsmith import snr_db
    per_tensor = quantize_tensor(outlier_col_tensor, "int8-per-tensor").dequantize()
    per_channel = quantize_tensor(outlier_col_tensor, "int8-per-channel").dequantize()
    assert snr_db(outlier_col_tensor, per_channel) > snr_db(outlier_col_tensor, per_tensor)


# ---------- argmax agreement ----------
def test_argmax_agreement_identical_inputs():
    from weightsmith import argmax_agreement
    one_d = [0.3, -1.2, 0.9, 0.1]
    two_d = [[0.3, -1.2, 0.9], [0.0, 5.0, -5.0], [1.0, 1.0, -1.0]]
    assert argmax_agreement(one_d, one_d) == 1.0
    assert argmax_agreement(two_d, two_d) == 1.0


def test_argmax_drops_after_aggressive_quantization(outlier_row_tensor):
    from weightsmith import argmax_agreement
    restored = quantize_tensor(outlier_row_tensor, "int8-per-tensor").dequantize()
    assert argmax_agreement(outlier_row_tensor, restored) < 1.0


def test_argmax_ties_break_to_lowest_index():
    from weightsmith import argmax_agreement
    assert argmax_agreement([1.0, 1.0], [1.0, 0.9]) == 1.0   # both argmax → index 0
    assert argmax_agreement([1.0, 3.0], [3.0, 1.0]) == 0.0


# ---------- size accounting ----------
def test_size_accounting_row8_is_about_half_of_fp16():
    big = [lcg_floats(9 + r, 64, -1.0, 1.0) for r in range(64)]  # 64x64
    fp16 = quantize_tensor(big, "fp16")
    per_tensor = quantize_tensor(big, "int8-per-tensor")
    per_row = quantize_tensor(big, "int8-per-row")
    per_channel = quantize_tensor(big, "int8-per-channel")
    assert fp16.size_bytes == 2 * 64 * 64            # 2 bytes per element
    assert per_tensor.size_bytes == 64 * 64 + 2      # 1 byte per element + 2
    assert per_row.size_bytes == 64 * 64 + 2 * 64    # + one f16 scale per row
    assert per_channel.size_bytes == 64 * 64 + 2 * 64
    assert 0.5 < per_row.size_bytes / fp16.size_bytes <= 0.53  # ≈ half of fp16


# ---------- error handling ----------
def test_ragged_rows_raise():
    with pytest.raises(QuantError, match="ragged"):
        quantize_tensor([[1.0, 2.0, 3.0], [4.0, 5.0]], "int8-per-row")


def test_empty_tensors_raise():
    for values in ([], [[]], [[], []]):
        with pytest.raises(QuantError):
            quantize_tensor(values, "fp16")


def test_unknown_mode_raises():
    with pytest.raises(QuantError, match="unknown mode"):
        quantize_tensor([1.0, 2.0], "int4")
    with pytest.raises(QuantError):
        QuantizedTensor(shape=(2,), mode="int4", data=[1, 2], scales=[])


def test_non_numeric_non_finite_and_3d_raise():
    with pytest.raises(QuantError):
        quantize_tensor([1.0, float("nan")], "fp16")
    with pytest.raises(QuantError):
        quantize_tensor([1.0, float("inf")], "int8-per-tensor")
    with pytest.raises(QuantError):
        quantize_tensor([1.0, "2.0"], "fp16")
    with pytest.raises(QuantError):
        quantize_tensor([[[1.0]]], "fp16")  # 3D is out of scope


# ---------- 1D semantics ----------
def test_one_d_tensor_is_a_single_row():
    per_row = quantize_tensor(ONE_D, "int8-per-row")
    per_channel = quantize_tensor(ONE_D, "int8-per-channel")
    per_tensor = quantize_tensor(ONE_D, "int8-per-tensor")
    assert per_row.shape == (8,)
    assert len(per_row.scales) == 1                    # single row → one scale
    assert per_row.size_bytes == per_tensor.size_bytes  # degenerates to per-tensor
    assert len(per_channel.scales) == 8                # one scale per element
    back = per_row.dequantize()
    assert not isinstance(back[0], list)               # 1D dequantizes flat
    assert all(abs(d - v) <= per_row.scales[0] / 2 + 1e-9 for d, v in zip(back, ONE_D))
    assert per_channel.size_bytes > quantize_tensor(ONE_D, "fp16").size_bytes  # wasteful on 1D
