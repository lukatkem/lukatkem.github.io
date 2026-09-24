"""Command-line interface: demo, check.

Both commands are fully deterministic — every synthetic tensor comes from a
fixed linear congruential generator (no `random` module), so the same
command prints the same numbers on every run.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from .calibrate import calibrate_tensor
from .errors import QuantError, WeightsmithError
from .metrics import argmax_agreement, max_abs_error, mean_abs_error, snr_db
from .pipeline import quantize_state_dict
from .quant import MODES, quantize_tensor

# Fixed LCG (MMIX multiplier/increment) — same seed, same tensor, every run.
_LCG_MULT = 6364136223846793005
_LCG_INC = 1442695040888963407
_LCG_MOD = 1 << 64
_LCG_SEED = 20260924


def lcg_floats(seed: int, count: int, lo: float = -1.0, hi: float = 1.0) -> List[float]:
    """Deterministic pseudo-random floats in [lo, hi) from a fixed LCG."""
    state = seed & (_LCG_MOD - 1)
    out: List[float] = []
    for _ in range(count):
        state = (state * _LCG_MULT + _LCG_INC) % _LCG_MOD
        unit = (state >> 11) / float(1 << 53)  # top 53 bits → [0, 1)
        out.append(lo + (hi - lo) * unit)
    return out


def _grid(values: List[float], cols: int) -> List[List[float]]:
    return [values[i:i + cols] for i in range(0, len(values), cols)]


def _demo_tensors() -> List[Tuple[str, List[List[float]]]]:
    well = _grid(lcg_floats(11, 16 * 16, -0.6, 0.6), 16)
    spiked = [v * 40.0 if i % 16 == 11 else v
              for i, v in enumerate(lcg_floats(22, 16 * 16, -0.5, 0.5))]
    outlier_channel = _grid(spiked, 16)
    tiny = _grid(lcg_floats(33, 3 * 2, -1.0, 1.0), 2)
    return [("well-behaved", well), ("outlier-channel", outlier_channel), ("tiny", tiny)]


def _mode_table(values: List[List[float]]) -> None:
    fp16_bytes = 2 * len(values) * len(values[0])
    print(f"  {'mode':<17}{'snr dB':>9}{'agreement':>11}{'bytes':>8}{'vs fp16':>9}")
    for mode in MODES:
        qt = quantize_tensor(values, mode)
        restored = qt.dequantize()
        print(f"  {mode:<17}{snr_db(values, restored):>9.1f}"
              f"{argmax_agreement(values, restored):>11.3f}"
              f"{qt.size_bytes:>8}{qt.size_bytes / fp16_bytes:>8.2f}x")


def _demo(a: argparse.Namespace) -> int:
    print("== weightsmith demo — three tensors, every mode ==\n")
    tensors = _demo_tensors()
    for i, (name, values) in enumerate(tensors, 1):
        print(f"[{i}/{len(tensors)}] {name} · {len(values)}x{len(values[0])}")
        _mode_table(values)
        pick = calibrate_tensor(values)
        print(f"  → calibrate (floor 0.95) picks {pick.mode}\n")
    print("== pipeline — quantize_state_dict with calibration ==")
    quantized, report = quantize_state_dict(dict(tensors))
    for t in report.tensors:
        print(f"  {t.name:<17}{t.mode:<17} snr {t.snr_db:>6.1f} · agree {t.agreement:.3f} · "
              f"{t.original_bytes} B → {t.quantized_bytes} B")
    print(f"totals: {report.original_bytes} B → {report.quantized_bytes} B "
          f"({report.ratio:.2f}x smaller)\n")
    print("done — pure stdlib, fully deterministic.")
    return 0


def _check(a: argparse.Namespace) -> int:
    if a.rows < 1 or a.cols < 1:
        raise QuantError("--rows and --cols must be >= 1")
    values = _grid(lcg_floats(_LCG_SEED, a.rows * a.cols, -1.0, 1.0), a.cols)
    qt = quantize_tensor(values, a.mode)
    restored = qt.dequantize()
    fp16_bytes = 2 * a.rows * a.cols
    print(f"tensor: {a.rows}x{a.cols} · LCG seed {_LCG_SEED} (deterministic)")
    print(f"mode:   {qt.mode} · scales {len(qt.scales)} · dtype {qt.dtype}")
    print(f"max_abs_error:    {max_abs_error(values, restored):.6f}")
    print(f"mean_abs_error:   {mean_abs_error(values, restored):.6f}")
    print(f"snr_db:           {snr_db(values, restored):.2f}")
    print(f"argmax_agreement: {argmax_agreement(values, restored):.3f}")
    print(f"bytes: {fp16_bytes} (fp16) → {qt.size_bytes} ({qt.mode})  "
          f"{fp16_bytes / qt.size_bytes:.2f}x smaller")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="weightsmith",
                                 description="post-training weight quantization, from scratch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="three synthetic tensors through every mode")
    demo.set_defaults(func=_demo)

    ck = sub.add_parser("check", help="quantize a deterministic LCG tensor and print metrics")
    ck.add_argument("--rows", type=int, default=12)
    ck.add_argument("--cols", type=int, default=8)
    ck.add_argument("--mode", default="int8-per-row")
    ck.set_defaults(func=_check)

    args = ap.parse_args(list(sys.argv[1:]) if argv is None else list(argv))
    try:
        return args.func(args)
    except WeightsmithError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
