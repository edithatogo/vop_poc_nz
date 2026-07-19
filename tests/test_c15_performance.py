from __future__ import annotations

import math
from typing import cast

import pytest

from vop_poc_nz.c15_performance import (
    PerformanceRegression,
    confidence_interval,
    current_runner,
    normalized_runner_identity,
    performance_budget,
    performance_config_digest,
    performance_ratchet,
    runner_fingerprint,
)

_PARAMETERS = {
    "repeats": 9,
    "iterations": 40,
    "rows": 4096,
    "strategies": 4,
    "dtype": "float64",
}
_CONFIG_DIGEST = performance_config_digest(
    workload_id="c15-evpi-4096x4-f64-v1",
    parameters=_PARAMETERS,
    confidence=0.95,
)


def _runner(*, os_name: str = "Linux", python: str = "3.14.6") -> dict[str, str | int]:
    return {
        "os": os_name,
        "os_release": "volatile-release",
        "os_version": "volatile-version",
        "architecture": "x86_64",
        "python": python,
        "python_implementation": "CPython",
        "python_compiler": "volatile-compiler",
        "cpu_count": 4,
        "cpu_model": "volatile-model",
        "ci_runner_os": os_name,
        "ci_runner_arch": "X64",
        "ci_image_os": "ubuntu24" if os_name == "Linux" else "win25-vs2026",
        "ci_image_version": "volatile-version",
    }


def _measurement(*, confidence: float = 0.95) -> dict[str, object]:
    return {
        "interval_method": "deterministic_percentile_bootstrap",
        "confidence": confidence,
        "bootstrap_resamples": 10_000,
        "minimum_samples": 5,
        "config_digest": _CONFIG_DIGEST,
    }


def _baseline() -> dict[str, object]:
    approved = normalized_runner_identity(_runner())
    return {
        "schema_version": "1.1.0",
        "workload_id": "c15-evpi-4096x4-f64-v1",
        "parameters": _PARAMETERS,
        "source": {
            "commit": "a" * 40,
            "workflow_run_id": "123",
            "captured_at": "2026-07-20",
        },
        "measurement": _measurement(),
        "cohorts": {
            "Linux": {
                "approved_runner_identity": approved,
                "approved_runner_fingerprint": runner_fingerprint(approved),
                "reference_upper_seconds": 0.01,
                "maximum_regression_factor": 3.0,
                "absolute_ceiling_seconds": 0.02,
            }
        },
    }


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
        performance_ratchet(
            samples,
            maximum_upper_seconds=2.0,
            runner={"os": "Linux", "architecture": "x86_64", "python": "3.14.6"},
        )


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


def test_normalized_runner_ignores_volatile_github_host_fields() -> None:
    left = _runner()
    right = {**left, "cpu_count": 16, "cpu_model": "other", "ci_image_version": "new"}
    assert normalized_runner_identity(left) == normalized_runner_identity(right)


def test_performance_budget_is_derived_from_cohort_baseline_and_ceiling() -> None:
    baseline = _baseline()
    budget = performance_budget(
        baseline,
        runner=_runner(),
        measurement=_measurement(),
    )
    assert budget["maximum_upper_seconds"] == pytest.approx(0.02)
    report = performance_ratchet(
        [0.009, 0.01, 0.011, 0.01, 0.01],
        baseline=baseline,
        runner=_runner(),
        config_digest=_CONFIG_DIGEST,
    )
    assert report["baseline"] == budget


def test_performance_baseline_fails_closed_for_unknown_cohort() -> None:
    with pytest.raises(ValueError, match="no 'Darwin' cohort"):
        performance_budget(
            _baseline(),
            runner=_runner(os_name="Darwin"),
            measurement=_measurement(),
        )


def test_performance_baseline_requires_retained_source_evidence() -> None:
    with pytest.raises(ValueError, match="retained source evidence"):
        performance_budget(
            {"schema_version": "1.1.0", "cohorts": {}},
            runner=_runner(),
            measurement=_measurement(),
        )


def test_performance_baseline_rejects_runner_series_drift() -> None:
    with pytest.raises(ValueError, match="approved normalized cohort"):
        performance_budget(
            _baseline(),
            runner=_runner(python="3.15.0"),
            measurement=_measurement(),
        )


def test_performance_baseline_rejects_measurement_configuration_drift() -> None:
    with pytest.raises(ValueError, match="measurement configuration mismatch"):
        performance_budget(
            _baseline(),
            runner=_runner(),
            measurement=_measurement(confidence=0.99),
        )


def test_performance_baseline_rejects_tampered_runner_fingerprint() -> None:
    baseline = _baseline()
    cohorts = cast(dict[str, object], baseline["cohorts"])
    linux = cast(dict[str, object], cohorts["Linux"])
    linux["approved_runner_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint is inconsistent"):
        performance_budget(
            baseline,
            runner=_runner(),
            measurement=_measurement(),
        )


def test_performance_baseline_rejects_tampered_config_digest() -> None:
    baseline = _baseline()
    measurement = cast(dict[str, object], baseline["measurement"])
    measurement["config_digest"] = "0" * 64
    with pytest.raises(ValueError, match="config digest is inconsistent"):
        performance_budget(
            baseline,
            runner=_runner(),
            measurement=measurement,
        )
