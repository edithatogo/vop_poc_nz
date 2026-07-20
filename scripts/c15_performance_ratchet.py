#!/usr/bin/env python3
"""Measure a numerical reduction repeatedly and enforce its confidence budget."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vop_poc_nz.c15_performance import (
    PerformanceRegression,
    performance_config_digest,
    performance_ratchet,
)
from vop_poc_nz.c15_scientific_oracles import numpy_evpi

_WORKLOAD_ID = "c15-evpi-4096x4-f64-v1"


def _write(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


def measure(*, repeats: int, iterations: int) -> list[float]:
    """Return repeated wall times for a deterministic higher-dimensional EVPI kernel."""
    if repeats < 5 or iterations <= 0:
        raise ValueError("repeats must be at least 5 and iterations must be positive")
    values = np.linspace(-5000.0, 5000.0, 4096 * 4, dtype=np.float64).reshape(4096, 4)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        checksum = 0.0
        for _iteration in range(iterations):
            checksum += numpy_evpi(values)
        elapsed = time.perf_counter() - started
        if not np.isfinite(checksum):
            raise RuntimeError("performance workload produced a non-finite result")
        samples.append(elapsed)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/c15_performance_baseline.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        baseline: Any = json.loads(args.baseline.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise ValueError("performance baseline must be a JSON object")
        if baseline.get("workload_id") != _WORKLOAD_ID:
            raise ValueError("performance baseline workload identity mismatch")
        parameters = baseline.get("parameters")
        expected_parameters = {
            "repeats": args.repeats,
            "iterations": args.iterations,
            "rows": 4096,
            "strategies": 4,
            "dtype": "float64",
        }
        if parameters != expected_parameters:
            raise ValueError(
                "performance baseline parameters do not match the workload"
            )
        config_digest = performance_config_digest(
            workload_id=_WORKLOAD_ID,
            parameters=expected_parameters,
            confidence=args.confidence,
        )
        samples = measure(repeats=args.repeats, iterations=args.iterations)
        report = performance_ratchet(
            samples,
            baseline=baseline,
            confidence=args.confidence,
            config_digest=config_digest,
        )
    except (OSError, PerformanceRegression, RuntimeError, ValueError) as exc:
        _write(
            args.output,
            {
                "schema_version": "1.0.0",
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "baseline": str(args.baseline),
                "parameters": {
                    "repeats": args.repeats,
                    "iterations": args.iterations,
                    "confidence": args.confidence,
                },
            },
        )
        print(f"C15 performance assurance failed: {exc}")
        return 2
    _write(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
