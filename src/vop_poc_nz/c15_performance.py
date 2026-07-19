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
    samples: list[float], *, confidence: float = 0.95, bootstrap_resamples: int = 10_000
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
        generator = random.Random(0xC15)
        size = len(samples)
        means = sorted(
            statistics.fmean(samples[generator.randrange(size)] for _ in range(size))
            for _ in range(bootstrap_resamples)
        )
        tail = (1.0 - confidence) / 2.0
        lower = _percentile(means, tail)
        upper = _percentile(means, 1.0 - tail)
        method = "deterministic_percentile_bootstrap"
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


def runner_fingerprint(runner: Mapping[str, str | int]) -> str:
    """Hash a canonical runner identity without unstable host names or identifiers."""
    if not runner or any(not str(key).strip() for key in runner):
        raise ValueError("runner identity must not be empty")
    canonical = json.dumps(dict(runner), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def performance_ratchet(
    samples: list[float],
    *,
    maximum_upper_seconds: float | None = None,
    baseline: Mapping[str, object] | None = None,
    confidence: float = 0.95,
    runner: Mapping[str, str | int] | None = None,
    minimum_samples: int = 5,
) -> dict[str, object]:
    """Require the upper confidence bound to remain within an absolute budget."""
    if len(samples) < minimum_samples:
        raise ValueError(f"at least {minimum_samples} performance samples are required")
    identity = dict(runner or current_runner())
    baseline_evidence: dict[str, object] | None = None
    if baseline is not None:
        if maximum_upper_seconds is not None:
            raise ValueError("provide a baseline or an absolute budget, not both")
        baseline_evidence = performance_budget(baseline, runner_os=str(identity["os"]))
        baseline_limit = baseline_evidence["maximum_upper_seconds"]
        if not isinstance(baseline_limit, float):
            raise ValueError("derived performance budget must be a float")
        maximum_upper_seconds = baseline_limit
    if maximum_upper_seconds is None:
        raise ValueError("a performance baseline or absolute budget is required")
    if not math.isfinite(maximum_upper_seconds) or maximum_upper_seconds <= 0.0:
        raise ValueError("maximum upper confidence bound must be positive and finite")
    interval = confidence_interval(samples, confidence=confidence)
    if interval.upper > maximum_upper_seconds:
        raise PerformanceRegression(
            f"upper confidence bound {interval.upper:.9f}s exceeds "
            f"budget {maximum_upper_seconds:.9f}s"
        )
    return {
        "schema_version": "1.0.0",
        "runner": identity,
        "runner_fingerprint": runner_fingerprint(identity),
        "metric": "wall_seconds",
        "interval": asdict(interval),
        "maximum_upper_seconds": maximum_upper_seconds,
        "passed": True,
        "samples": samples,
        "baseline": baseline_evidence,
    }


def performance_budget(
    baseline: Mapping[str, object], *, runner_os: str
) -> dict[str, object]:
    """Derive a bounded cohort budget from retained exact-head observations."""
    if baseline.get("schema_version") != "1.0.0":
        raise ValueError("unsupported performance baseline schema")
    source = baseline.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("performance baseline requires retained source evidence")

    def has_source_text(field: str) -> bool:
        value = source.get(field)
        return isinstance(value, str) and bool(value.strip())

    if not all(
        has_source_text(field) for field in ("commit", "workflow_run_id", "captured_at")
    ):
        raise ValueError("performance baseline requires retained source evidence")
    cohorts = baseline.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise ValueError("performance baseline requires cohorts")
    cohort = cohorts.get(runner_os)
    if not isinstance(cohort, Mapping):
        raise ValueError(f"performance baseline has no {runner_os!r} cohort")

    def positive(field: str) -> float:
        value = cohort.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"baseline {field} must be numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"baseline {field} must be positive and finite")
        return result

    reference = positive("reference_upper_seconds")
    factor = positive("maximum_regression_factor")
    ceiling = positive("absolute_ceiling_seconds")
    return {
        "source": dict(source),
        "runner_os": runner_os,
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
    "performance_budget",
    "performance_ratchet",
    "runner_fingerprint",
]
