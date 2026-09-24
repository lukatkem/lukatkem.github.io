# weightsmith — post-training weight quantization from scratch

**Shrink a model 2× with provable quality — every quantization scheme from scratch.**

## Why this matters

Every deployed LLM ships its weights quantized, yet most developers treat the
int8 checkpoint as a black box. This is the whole mechanism — symmetric scale
selection, per-tensor / per-row / per-channel granularity, quality metrics,
and a calibrator — understood at the level where the error bound is provable
(quantization error ≤ scale/2), not imported. The approach behind `int8-per-row`
here is the same one that halved a real browser-demo model download at 100%
output agreement.

## Quickstart

```bash
python -m pytest -q                        # 40 tests
python -m weightsmith demo                 # three tensors through every mode
python -m weightsmith check --rows 16 --cols 16 --mode int8-per-row
```

`check` synthesizes a deterministic tensor from a fixed LCG (no `random`
module) and prints max/mean error, snr, argmax agreement, and bytes — the
same command always prints the same numbers.

## The modes

All schemes are symmetric (`scale = absmax / 127`, so zero maps to zero and
the payload always fits in [-127, 127]):

| mode | scales stored | bytes on disk | error bound | best for |
|---|---|---|---|---|
| `fp16` | none (exact baseline) | 2 per element | 0 | round-trip reference, tight floors |
| `int8-per-tensor` | 1 | 1/elem + 2 | scale/2 | uniform tensors, smallest overhead |
| `int8-per-row` | 1 per row | 1/elem + 2·rows | row scale/2 | outlier rows — the 2× shipped win |
| `int8-per-channel` | 1 per column | 1/elem + 2·cols | col scale/2 | outlier columns/channels |

A 1D tensor is treated as a single row: per-row collapses to one scale,
per-channel to one scale per element (exact but wasteful, so calibration
never picks it).

## API

| call | what it does |
|---|---|
| `quantize_tensor(values, mode)` | 1D/2D lists → `QuantizedTensor`; ragged/empty/unknown modes raise `QuantError` |
| `QuantizedTensor.dequantize()` | one multiply per element; fp16 round-trips exactly |
| `.size_bytes` / `.dtype` | disk accounting (int8 = 1 B/value + 2 B/scale; fp16 = 2 B/value) |
| `snr_db`, `max_abs_error`, `mean_abs_error` | pure-math quality metrics |
| `argmax_agreement(a, b)` | fraction of rows whose max lands on the same index — the "same prediction" metric |
| `calibrate_tensor(values, min_agreement=0.95)` | measures every mode, returns the smallest that clears the floor; fp16 is the graceful fallback |
| `quantize_state_dict(state, mode=None)` | whole model in → dict of `QuantizedTensor` + `Report` (per-tensor quality, byte totals, ratio) |

## Guarantees

- **Provable error bound** — every int8 value satisfies `|dequant − original| ≤ scale/2`; the tests check it element-by-element.
- **Deterministic** — fixed LCG seeds, round-half-away-from-zero, tie-breaks to the lowest index. No `random`, no numpy — pure stdlib lists, ints, floats.
- **Honest accounting** — `size_bytes` counts real bytes on disk; the report's ratio is measured against the fp16 baseline, not a fantasy number.

## Honest scope

Post-training **weight** quantization only: no activation calibration, no
GPTQ-style error compensation, no packed bit layouts or kernels — dequantize
is a multiply and that is the whole runtime story. Per-row int8 on a real
model is the proven core; everything else here is the generalization around
it. Part of an eight-project from-scratch AI systems portfolio.
