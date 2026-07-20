#!/usr/bin/env python3
"""Compare a pinned Scalene JSON profile with a committed portable baseline."""

from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

from vop_poc_nz.assurance_policy import exceeds_upper_bound


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return normalized


def scalene_profile_report(
    profile: dict[str, object],
    baseline: dict[str, object],
    *,
    scalene_version: str,
) -> dict[str, object]:
    """Return a fail-closed regression report over portable aggregate metrics."""
    required_version = baseline.get("required_scalene_version")
    if not isinstance(required_version, str):
        raise ValueError("baseline requires required_scalene_version")
    workload = baseline.get("workload")
    if not isinstance(workload, str) or not workload:
        raise ValueError("baseline requires workload")
    program = profile.get("program")
    if not isinstance(program, str) or Path(program).name != workload:
        raise ValueError("Scalene profile workload identity mismatch")
    elapsed = _finite_number(profile.get("elapsed_time_sec"), field="elapsed_time_sec")
    reference = _finite_number(
        baseline.get("reference_elapsed_seconds"),
        field="reference_elapsed_seconds",
    )
    maximum_elapsed = _finite_number(
        baseline.get("maximum_elapsed_seconds"),
        field="maximum_elapsed_seconds",
    )
    maximum_factor = _finite_number(
        baseline.get("maximum_regression_factor"),
        field="maximum_regression_factor",
    )
    if reference == 0.0 or maximum_elapsed == 0.0 or maximum_factor == 0.0:
        raise ValueError("Scalene baseline limits must be greater than zero")
    profile_files = profile.get("files")
    if not isinstance(profile_files, dict) or not profile_files:
        raise ValueError("Scalene profile requires at least one profiled file")
    factor = elapsed / reference
    failures: list[str] = []
    if scalene_version != required_version:
        failures.append(
            f"scalene_version: {scalene_version} != required {required_version}"
        )
    if exceeds_upper_bound(elapsed, maximum_elapsed):
        failures.append(
            f"elapsed_time_sec: {elapsed:.6f} exceeds {maximum_elapsed:.6f}"
        )
    if exceeds_upper_bound(factor, maximum_factor):
        failures.append(
            f"elapsed_regression_factor: {factor:.3f} exceeds {maximum_factor:.3f}"
        )
    return {
        "schema_version": "1.0.0",
        "workload": workload,
        "scalene_version": scalene_version,
        "reference_environment": baseline.get("reference_environment"),
        "reference_elapsed_seconds": reference,
        "elapsed_time_seconds": elapsed,
        "elapsed_regression_factor": factor,
        "profiled_file_count": len(profile_files),
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/scalene_profile_baseline.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".benchmarks/scalene-regression.json")
    )
    args = parser.parse_args()
    raw_profile: Any = json.loads(args.profile.read_text(encoding="utf-8"))
    raw_baseline: Any = json.loads(args.baseline.read_text(encoding="utf-8"))
    if not isinstance(raw_profile, dict) or not isinstance(raw_baseline, dict):
        raise ValueError("Scalene profile and baseline must be JSON objects")
    report = scalene_profile_report(
        raw_profile,
        raw_baseline,
        scalene_version=version("scalene"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
