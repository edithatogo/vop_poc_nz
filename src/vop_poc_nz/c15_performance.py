"""Runner-scoped repeated-sample performance confidence ratchets."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import NormalDist


class PerformanceRegression(RuntimeError):
    """Raised when a confidence-bound performance budget is exceeded."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Two-sided normal confidence interval for repeated measurements."""

    confidence: float
    samples: int
    mean: float
    standard_deviation: float
    standard_error: float
    lower: float
    upper: float


def confidence_interval(
    samples: list[float], *, confidence: float = 0.95
) -> ConfidenceInterval:
    """Calculate a deterministic two-sided interval from finite positive samples."""
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("performance samples must be finite and non-negative")
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples)
    error = deviation / math.sqrt(len(samples))
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = critical * error
    return ConfidenceInterval(
        confidence=confidence,
        samples=len(samples),
        mean=mean,
        standard_deviation=deviation,
        standard_error=error,
        lower=max(0.0, mean - margin),
        upper=mean + margin,
    )


def current_runner() -> dict[str, str | int]:
    """Return the bounded runtime properties that define a comparable runner."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count() or 1,
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
    maximum_upper_seconds: float,
    confidence: float = 0.95,
    runner: Mapping[str, str | int] | None = None,
    minimum_samples: int = 5,
) -> dict[str, object]:
    """Require the upper confidence bound to remain within an absolute budget."""
    if len(samples) < minimum_samples:
        raise ValueError(f"at least {minimum_samples} performance samples are required")
    if not math.isfinite(maximum_upper_seconds) or maximum_upper_seconds <= 0.0:
        raise ValueError("maximum upper confidence bound must be positive and finite")
    interval = confidence_interval(samples, confidence=confidence)
    identity = dict(runner or current_runner())
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
    }


__all__ = [
    "ConfidenceInterval",
    "PerformanceRegression",
    "confidence_interval",
    "current_runner",
    "performance_ratchet",
    "runner_fingerprint",
]
