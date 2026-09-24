"""CLI — deterministic LCG, reproducible check output, demo end-to-end."""
from __future__ import annotations

from weightsmith.cli import _LCG_SEED, lcg_floats, main


# ---------- LCG ----------
def test_lcg_is_deterministic_and_bounded():
    a = lcg_floats(_LCG_SEED, 100)
    b = lcg_floats(_LCG_SEED, 100)
    assert a == b                                  # same seed, same values, every time
    assert all(-1.0 <= v < 1.0 for v in a)
    assert lcg_floats(_LCG_SEED + 1, 100) != a     # different seed, different stream
    assert lcg_floats(7, 5, 0.0, 10.0) != lcg_floats(8, 5, 0.0, 10.0)


# ---------- check ----------
def test_check_is_reproducible(capsys):
    argv = ["check", "--rows", "6", "--cols", "4", "--mode", "int8-per-row"]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    second = capsys.readouterr().out
    assert first == second                         # fully deterministic
    assert "snr_db" in first and "argmax_agreement" in first
    assert "int8-per-row" in first and "max_abs_error" in first


def test_check_mode_and_shape_change_output(capsys):
    main(["check", "--rows", "6", "--cols", "4", "--mode", "int8-per-row"])
    small = capsys.readouterr().out
    main(["check", "--rows", "6", "--cols", "4", "--mode", "fp16"])
    fp16 = capsys.readouterr().out
    main(["check", "--rows", "8", "--cols", "4", "--mode", "int8-per-row"])
    bigger = capsys.readouterr().out
    assert fp16 != small                           # mode changes the numbers
    assert bigger != small                         # shape changes the numbers
    assert "6x4" in small and "8x4" in bigger


def test_check_unknown_mode_exits_1(capsys):
    assert main(["check", "--mode", "int4"]) == 1
    assert "error:" in capsys.readouterr().err


def test_check_rejects_nonpositive_dimensions(capsys):
    assert main(["check", "--rows", "0", "--cols", "4"]) == 1
    assert "error:" in capsys.readouterr().err


# ---------- demo ----------
def test_demo_runs_end_to_end(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    for tensor in ("well-behaved", "outlier-channel", "tiny"):
        assert tensor in out
    for mode in ("fp16", "int8-per-tensor", "int8-per-row", "int8-per-channel"):
        assert mode in out                         # every mode, every tensor
    assert "agreement" in out and "totals" in out and "deterministic" in out
    assert out.count("calibrate") == 3             # a pick per tensor
