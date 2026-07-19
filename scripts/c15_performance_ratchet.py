#!/usr/bin/env python3
"""Measure a numerical reduction repeatedly and enforce its confidence budget."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vop_poc_nz.c15_performance import (
    PerformanceRegression,
    performance_ratchet,
)
from vop_poc_nz.c15_scientific_oracles import numpy_evpi


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
    parser.add_argument("--maximum-upper-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        samples = measure(repeats=args.repeats, iterations=args.iterations)
        report = performance_ratchet(
            samples,
            maximum_upper_seconds=args.maximum_upper_seconds,
            confidence=args.confidence,
        )
    except (PerformanceRegression, RuntimeError, ValueError) as exc:
        print(f"C15 performance assurance failed: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
