#!/usr/bin/env python3
"""Benchmark Arrow interchange and enforce conservative regression budgets."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from vop_poc_nz.perspective_io import write_ipc_records, write_records


def _elapsed(operation: Callable[[], object], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def benchmark_serialization(rows: int = 20_000, repeats: int = 3) -> dict[str, object]:
    """Measure formats and report hardware-normalised ratios against JSON Lines."""
    records = [
        {
            "draw": str(index),
            "strategy": "intervention" if index % 2 else "usual-care",
            "perspective": "societal",
            "net_benefit": float(index) / 7,
        }
        for index in range(rows)
    ]
    with tempfile.TemporaryDirectory(prefix="vop-serialization-") as directory:
        root = Path(directory)
        timings = {
            "jsonl_write_seconds": _elapsed(
                lambda: write_records(
                    records, root / "records.jsonl", prefer_parquet=False
                ),
                repeats,
            ),
            "parquet_write_seconds": _elapsed(
                lambda: write_records(records, root / "records.parquet"), repeats
            ),
            "ipc_write_seconds": _elapsed(
                lambda: write_ipc_records(records, root / "records.arrow"), repeats
            ),
        }
    reference = timings["jsonl_write_seconds"]
    ratios = {
        "parquet_write_vs_jsonl": timings["parquet_write_seconds"] / reference,
        "ipc_write_vs_jsonl": timings["ipc_write_seconds"] / reference,
    }
    return {
        "schema_version": "1.0.0",
        "rows": rows,
        "repeats": repeats,
        "timings": timings,
        "ratios": ratios,
    }


def regression_failures(
    result: dict[str, object], budgets: dict[str, float]
) -> list[str]:
    """Return ratios exceeding explicit, versioned regression budgets."""
    ratios = result["ratios"]
    assert isinstance(ratios, dict)
    return [
        f"{name}: {float(ratios[name]):.3f} exceeds {limit:.3f}"
        for name, limit in budgets.items()
        if float(ratios[name]) > limit
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--budgets", type=Path, default=Path("benchmarks/serialization_budgets.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark_serialization(args.rows, args.repeats)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    budgets = json.loads(args.budgets.read_text(encoding="utf-8"))["maximum_ratios"]
    failures = regression_failures(result, budgets)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
