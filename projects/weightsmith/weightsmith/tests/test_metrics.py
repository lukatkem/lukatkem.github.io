"""Metrics — exact values on known inputs, plus input validation."""
from __future__ import annotations

import math

import pytest

from weightsmith import MetricsError, argmax_agreement, max_abs_error, mean_abs_error, snr_db


def test_max_and_mean_abs_error_exact():
    a = [[1.0, -2.0, 0.5]]
    b = [[0.5, -1.0, 0.5]]
    assert max_abs_error(a, b) == 1.0
    assert mean_abs_error(a, b) == pytest.approx(1.5 / 3)


def test_snr_db_known_value():
    # signal power 1, noise power 0.25 → 10·log10(4)
    assert snr_db([1.0, 0.0], [0.5, 0.0]) == pytest.approx(10 * math.log10(4))


def test_snr_db_identical_inputs_is_infinite():
    assert math.isinf(snr_db([0.1, -0.2, 0.3], [0.1, -0.2, 0.3]))
    assert math.isinf(snr_db([[1.0], [-2.0]], [[1.0], [-2.0]]))


def test_snr_db_orders_quality():
    a = [1.0, -1.0, 0.5, -0.5]
    close = [v + 0.01 for v in a]
    coarse = [v + 0.2 for v in a]
    assert snr_db(a, close) > snr_db(a, coarse)


def test_argmax_agreement_partial():
    a = [[1.0, 3.0, 2.0], [0.0, -1.0, 5.0]]
    assert argmax_agreement(a, a) == 1.0
    assert argmax_agreement(a, [[1.0, 3.0, 2.0], [0.0, 6.0, 5.0]]) == 0.5  # row 1 flipped
    assert argmax_agreement(a, [[3.0, 2.0, 1.0], [0.0, 6.0, 5.0]]) == 0.0  # both rows flipped


def test_metrics_reject_empty_ragged_mismatched_and_non_finite():
    with pytest.raises(MetricsError):
        max_abs_error([], [])
    with pytest.raises(MetricsError):
        mean_abs_error([[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0]])     # ragged
    with pytest.raises(MetricsError):
        snr_db([1.0, 2.0], [1.0, 2.0, 3.0])                          # length mismatch
    with pytest.raises(MetricsError):
        argmax_agreement([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])     # row-count mismatch
    with pytest.raises(MetricsError):
        snr_db([1.0, float("nan")], [1.0, 0.0])                      # non-finite
