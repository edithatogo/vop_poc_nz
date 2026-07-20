"""Runner-scoped repeated-sample performance confidence ratchets."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import NormalDist
from typing import cast

_BOOTSTRAP_METHOD = "deterministic_percentile_bootstrap"
_DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


class PerformanceRegression(RuntimeError):
    """Raised when a confidence-bound performance budget is exceeded."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Two-sided confidence interval for repeated measurements."""

    method: str
    confidence: float
    samples: int
    resamples: int
    mean: float
    standard_deviation: float
    standard_error: float
    lower: float
    upper: float


def confidence_interval(
    samples: list[float],
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
) -> ConfidenceInterval:
    """Calculate a deterministic percentile-bootstrap interval when supported."""
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("performance samples must be finite and non-negative")
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples)
    error = deviation / math.sqrt(len(samples))
    if len(samples) >= 5:
        if bootstrap_resamples < 1_000:
            raise ValueError("bootstrap_resamples must be at least 1000")
        # This PRNG is intentionally reproducible statistical resampling, not security.
        generator = random.Random(0xC15)  # nosec B311
        size = len(samples)
        means = sorted(
            statistics.fmean(samples[generator.randrange(size)] for _ in range(size))
            for _ in range(bootstrap_resamples)
        )
        tail = (1.0 - confidence) / 2.0
        lower = _percentile(means, tail)
        upper = _percentile(means, 1.0 - tail)
        method = _BOOTSTRAP_METHOD
        resamples = bootstrap_resamples
    else:
        critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        margin = critical * error
        lower = max(0.0, mean - margin)
        upper = mean + margin
        method = "normal"
        resamples = 0
    return ConfidenceInterval(
        method=method,
        confidence=confidence,
        samples=len(samples),
        resamples=resamples,
        mean=mean,
        standard_deviation=deviation,
        standard_error=error,
        lower=max(0.0, lower),
        upper=upper,
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile from sorted finite values."""
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _cpu_model() -> str:
    """Return a stable hardware cohort label without host-specific identifiers."""
    processor = platform.processor().strip()
    if processor:
        return processor
    windows_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if windows_identifier:
        return windows_identifier
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.casefold().startswith("model name"):
                    return line.partition(":")[2].strip()
    except OSError:
        pass
    return "unknown"


def _environment_value(name: str) -> str:
    """Read runner-provided mixed-case variables without normalizing their names."""
    return os.environ.get(name, "local")


def current_runner() -> dict[str, str | int]:
    """Return the bounded runtime properties that define a comparable runner."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "cpu_count": os.cpu_count() or 1,
        "cpu_model": _cpu_model(),
        "ci_runner_os": os.environ.get("RUNNER_OS", "local"),
        "ci_runner_arch": os.environ.get("RUNNER_ARCH", "local"),
        "ci_image_os": _environment_value("ImageOS"),
        "ci_image_version": _environment_value("ImageVersion"),
    }


def _normalized_architecture(value: object) -> str:
    architecture = str(value).strip().casefold().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }
    return aliases.get(architecture, architecture)


def normalized_runner_identity(
    runner: Mapping[str, str | int],
) -> dict[str, str]:
    """Return stable GitHub runner cohort fields, excluding volatile host details."""
    required = (
        "os",
        "architecture",
        "python",
        "python_implementation",
        "ci_runner_os",
        "ci_runner_arch",
        "ci_image_os",
    )
    missing = [field for field in required if not str(runner.get(field, "")).strip()]
    if missing:
        raise ValueError(
            "runner identity is missing normalized cohort fields: " + ", ".join(missing)
        )
    python_parts = str(runner["python"]).strip().split(".")
    if len(python_parts) < 2 or not all(part.isdigit() for part in python_parts[:2]):
        raise ValueError(
            "runner python version must include a numeric major.minor series"
        )
    return {
        "os": str(runner["os"]).strip(),
        "architecture": _normalized_architecture(runner["architecture"]),
        "python_series": ".".join(python_parts[:2]),
        "python_implementation": str(runner["python_implementation"]).strip(),
        "ci_runner_os": str(runner["ci_runner_os"]).strip(),
        "ci_runner_arch": _normalized_architecture(runner["ci_runner_arch"]),
        "ci_image_os": str(runner["ci_image_os"]).strip().casefold(),
    }


def runner_fingerprint(runner: Mapping[str, str | int]) -> str:
    """Hash a canonical runner identity without unstable host names or identifiers."""
    if not runner or any(not str(key).strip() for key in runner):
        raise ValueError("runner identity must not be empty")
    canonical = json.dumps(dict(runner), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def performance_config_digest(
    *,
    workload_id: str,
    parameters: Mapping[str, object],
    confidence: float,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    minimum_samples: int = 5,
) -> str:
    """Hash the workload and confidence-method configuration used by a baseline."""
    if not workload_id.strip() or not parameters:
        raise ValueError("performance configuration requires workload identity")
    configuration = {
        "workload_id": workload_id,
        "parameters": dict(parameters),
        "interval_method": _BOOTSTRAP_METHOD,
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "minimum_samples": minimum_samples,
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _report_normalized_runner(
    runner: Mapping[str, str | int], *, required: bool
) -> dict[str, str] | None:
    try:
        return normalized_runner_identity(runner)
    except ValueError:
        if required:
            raise
        return None


def performance_ratchet(
    samples: list[float],
    *,
    maximum_upper_seconds: float | None = None,
    baseline: Mapping[str, object] | None = None,
    confidence: float = 0.95,
    runner: Mapping[str, str | int] | None = None,
    minimum_samples: int = 5,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    config_digest: str | None = None,
) -> dict[str, object]:
    """Require the upper confidence bound to remain within an absolute budget."""
    if len(samples) < minimum_samples:
        raise ValueError(f"at least {minimum_samples} performance samples are required")
    identity = dict(runner or current_runner())
    normalized_identity = _report_normalized_runner(
        identity, required=baseline is not None
    )
    measurement = {
        "interval_method": _BOOTSTRAP_METHOD,
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "minimum_samples": minimum_samples,
        "config_digest": config_digest,
    }
    baseline_evidence: dict[str, object] | None = None
    if baseline is not None:
        if maximum_upper_seconds is not None:
            raise ValueError("provide a baseline or an absolute budget, not both")
        if config_digest is None:
            raise ValueError("a performance config digest is required with a baseline")
        baseline_evidence = performance_budget(
            baseline,
            runner=identity,
            measurement=measurement,
        )
        baseline_limit = baseline_evidence["maximum_upper_seconds"]
        if not isinstance(baseline_limit, float):
            raise ValueError("derived performance budget must be a float")
        maximum_upper_seconds = baseline_limit
    if maximum_upper_seconds is None:
        raise ValueError("a performance baseline or absolute budget is required")
    if not math.isfinite(maximum_upper_seconds) or maximum_upper_seconds <= 0.0:
        raise ValueError("maximum upper confidence bound must be positive and finite")
    interval = confidence_interval(
        samples,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
    )
    if interval.upper > maximum_upper_seconds:
        raise PerformanceRegression(
            f"upper confidence bound {interval.upper:.9f}s exceeds "
            f"budget {maximum_upper_seconds:.9f}s"
        )
    report: dict[str, object] = {
        "schema_version": "1.1.0",
        "runner": identity,
        "runner_fingerprint": runner_fingerprint(identity),
        "metric": "wall_seconds",
        "interval": asdict(interval),
        "maximum_upper_seconds": maximum_upper_seconds,
        "passed": True,
        "samples": samples,
        "baseline": baseline_evidence,
    }
    if normalized_identity is not None:
        report.update(
            {
                "normalized_runner": normalized_identity,
                "normalized_runner_fingerprint": runner_fingerprint(
                    normalized_identity
                ),
            }
        )
    return report


def _retained_source(baseline: Mapping[str, object]) -> dict[str, object]:
    source = baseline.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("performance baseline requires retained source evidence")
    source_mapping = cast(Mapping[str, object], source)
    required = ("commit", "workflow_run_id", "captured_at")
    values = [source_mapping.get(field) for field in required]
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("performance baseline requires retained source evidence")
    return dict(source_mapping)


def _approved_cohort(
    baseline: Mapping[str, object], runner: Mapping[str, str | int]
) -> tuple[str, Mapping[str, object], dict[str, str], str]:
    cohorts = baseline.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise ValueError("performance baseline requires cohorts")
    cohort_mapping = cast(Mapping[str, object], cohorts)
    normalized_identity = normalized_runner_identity(runner)
    runner_os = normalized_identity["os"]
    cohort = cohort_mapping.get(runner_os)
    if not isinstance(cohort, Mapping):
        raise ValueError(f"performance baseline has no {runner_os!r} cohort")
    cohort = cast(Mapping[str, object], cohort)
    approved_identity = cohort.get("approved_runner_identity")
    approved_fingerprint = cohort.get("approved_runner_fingerprint")
    if not isinstance(approved_identity, Mapping) or not isinstance(
        approved_fingerprint, str
    ):
        raise ValueError("performance baseline requires an approved runner identity")
    approved_identity_dict = {
        str(key): str(value) for key, value in approved_identity.items()
    }
    if runner_fingerprint(approved_identity_dict) != approved_fingerprint:
        raise ValueError("performance baseline runner fingerprint is inconsistent")
    current_fingerprint = runner_fingerprint(normalized_identity)
    if (
        normalized_identity != approved_identity_dict
        or current_fingerprint != approved_fingerprint
    ):
        raise ValueError("current runner does not match the approved normalized cohort")
    return runner_os, cohort, approved_identity_dict, approved_fingerprint


def _positive_cohort_value(cohort: Mapping[str, object], field: str) -> float:
    value = cohort.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"baseline {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"baseline {field} must be positive and finite")
    return result


def _approved_measurement(baseline: Mapping[str, object]) -> dict[str, object]:
    measurement = baseline.get("measurement")
    workload_id = baseline.get("workload_id")
    parameters = baseline.get("parameters")
    if not isinstance(measurement, Mapping):
        raise ValueError("performance baseline requires measurement configuration")
    measurement = cast(Mapping[str, object], measurement)
    confidence = measurement.get("confidence")
    resamples = measurement.get("bootstrap_resamples")
    minimum_samples = measurement.get("minimum_samples")
    if (
        measurement.get("interval_method") != _BOOTSTRAP_METHOD
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or not isinstance(workload_id, str)
        or not isinstance(parameters, Mapping)
    ):
        raise ValueError("performance baseline measurement configuration is invalid")
    expected_digest = performance_config_digest(
        workload_id=workload_id,
        parameters=cast(Mapping[str, object], parameters),
        confidence=float(confidence),
        bootstrap_resamples=resamples,
        minimum_samples=minimum_samples,
    )
    if measurement.get("config_digest") != expected_digest:
        raise ValueError("performance baseline config digest is inconsistent")
    return dict(measurement)


def performance_budget(
    baseline: Mapping[str, object],
    *,
    runner: Mapping[str, str | int],
    measurement: Mapping[str, object],
) -> dict[str, object]:
    """Derive a bounded cohort budget from retained exact-head observations."""
    if baseline.get("schema_version") != "1.1.0":
        raise ValueError("unsupported performance baseline schema")
    source = _retained_source(baseline)
    runner_os, cohort, approved_identity, approved_fingerprint = _approved_cohort(
        baseline, runner
    )

    approved_measurement = _approved_measurement(baseline)
    if dict(measurement) != approved_measurement:
        raise ValueError("performance measurement configuration mismatch")

    reference = _positive_cohort_value(cohort, "reference_upper_seconds")
    factor = _positive_cohort_value(cohort, "maximum_regression_factor")
    ceiling = _positive_cohort_value(cohort, "absolute_ceiling_seconds")
    return {
        "source": source,
        "runner_os": runner_os,
        "approved_runner_identity": approved_identity,
        "approved_runner_fingerprint": approved_fingerprint,
        "measurement": approved_measurement,
        "reference_upper_seconds": reference,
        "maximum_regression_factor": factor,
        "absolute_ceiling_seconds": ceiling,
        "maximum_upper_seconds": min(reference * factor, ceiling),
    }


__all__ = [
    "ConfidenceInterval",
    "PerformanceRegression",
    "confidence_interval",
    "current_runner",
    "normalized_runner_identity",
    "performance_budget",
    "performance_config_digest",
    "performance_ratchet",
    "runner_fingerprint",
]
