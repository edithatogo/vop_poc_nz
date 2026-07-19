from __future__ import annotations

import math

import pytest

from vop_poc_nz.c15_performance import (
    PerformanceRegression,
    confidence_interval,
    current_runner,
    performance_budget,
    performance_ratchet,
    runner_fingerprint,
)


def test_confidence_interval_is_deterministic_and_contains_mean() -> None:
    interval = confidence_interval([0.9, 1.0, 1.1, 1.0, 1.0], confidence=0.95)
    assert interval.mean == pytest.approx(1.0)
    assert interval.lower <= interval.mean <= interval.upper
    assert interval.samples == 5
    assert interval.method == "deterministic_percentile_bootstrap"
    assert interval.resamples == 10_000
    assert interval == confidence_interval([0.9, 1.0, 1.1, 1.0, 1.0])


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


def test_current_runner_has_hardware_image_and_runtime_cohort_fields() -> None:
    assert {
        "os",
        "os_release",
        "os_version",
        "architecture",
        "python",
        "python_implementation",
        "python_compiler",
        "cpu_count",
        "cpu_model",
        "ci_runner_os",
        "ci_runner_arch",
        "ci_image_os",
        "ci_image_version",
    } <= current_runner().keys()


def test_performance_budget_is_derived_from_cohort_baseline_and_ceiling() -> None:
    baseline = {
        "schema_version": "1.0.0",
        "source": {
            "commit": "a" * 40,
            "workflow_run_id": "123",
            "captured_at": "2026-07-20",
        },
        "cohorts": {
            "Linux": {
                "reference_upper_seconds": 0.01,
                "maximum_regression_factor": 3.0,
                "absolute_ceiling_seconds": 0.02,
            }
        },
    }
    budget = performance_budget(baseline, runner_os="Linux")
    assert budget["maximum_upper_seconds"] == pytest.approx(0.02)
    report = performance_ratchet(
        [0.009, 0.01, 0.011, 0.01, 0.01],
        baseline=baseline,
        runner={"os": "Linux"},
    )
    assert report["baseline"] == budget


def test_performance_baseline_fails_closed_for_unknown_cohort() -> None:
    with pytest.raises(ValueError, match="no 'Darwin' cohort"):
        performance_budget(
            {
                "schema_version": "1.0.0",
                "source": {
                    "commit": "a" * 40,
                    "workflow_run_id": "123",
                    "captured_at": "2026-07-20",
                },
                "cohorts": {},
            },
            runner_os="Darwin",
        )


def test_performance_baseline_requires_retained_source_evidence() -> None:
    with pytest.raises(ValueError, match="retained source evidence"):
        performance_budget(
            {"schema_version": "1.0.0", "cohorts": {}},
            runner_os="Linux",
        )
