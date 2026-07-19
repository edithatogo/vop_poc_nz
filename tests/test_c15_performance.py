from __future__ import annotations

import math

import pytest

from vop_poc_nz.c15_performance import (
    PerformanceRegression,
    confidence_interval,
    performance_ratchet,
    runner_fingerprint,
)


def test_confidence_interval_is_deterministic_and_contains_mean() -> None:
    interval = confidence_interval([0.9, 1.0, 1.1, 1.0, 1.0], confidence=0.95)
    assert interval.mean == pytest.approx(1.0)
    assert interval.lower <= interval.mean <= interval.upper
    assert interval.samples == 5


def test_performance_ratchet_uses_upper_confidence_bound() -> None:
    report = performance_ratchet(
        [0.09, 0.10, 0.11, 0.10, 0.10, 0.09, 0.11],
        maximum_upper_seconds=0.2,
        runner={"os": "Linux", "architecture": "x86_64", "python": "3.14.6"},
    )
    assert report["passed"] is True
    assert report["runner_fingerprint"]
    with pytest.raises(PerformanceRegression, match="upper confidence bound"):
        performance_ratchet(
            [0.19, 0.20, 0.21, 0.20, 0.20],
            maximum_upper_seconds=0.15,
            runner={"os": "Linux", "architecture": "x86_64", "python": "3.14.6"},
        )


@pytest.mark.parametrize("samples", [[], [1.0], [1.0, math.inf, 1.0]])
def test_performance_ratchet_rejects_insufficient_or_invalid_samples(
    samples: list[float],
) -> None:
    with pytest.raises(ValueError):
        performance_ratchet(samples, maximum_upper_seconds=2.0)


def test_runner_fingerprint_is_order_independent() -> None:
    left = runner_fingerprint({"os": "Linux", "architecture": "x86_64"})
    right = runner_fingerprint({"architecture": "x86_64", "os": "Linux"})
    assert left == right
