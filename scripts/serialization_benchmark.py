#!/usr/bin/env python3
"""Benchmark Arrow interchange against deterministic resource envelopes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import psutil

from vop_poc_nz.perspective_io import write_ipc_records, write_records

BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def blas_thread_failures(environment: Mapping[str, str]) -> list[str]:
    """Return thread controls that do not enforce deterministic single-threading."""
    return [name for name in BLAS_THREAD_VARIABLES if environment.get(name) != "1"]


def _measure(operation: Callable[[], Path]) -> dict[str, float | int]:
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak_rss = [initial_rss]
    stopped = threading.Event()

    def sample_rss() -> None:
        while not stopped.wait(0.001):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, name="rss-sampler", daemon=True)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    sampler.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        output = operation()
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        current_traced, traced_peak = tracemalloc.get_traced_memory()
        after = tracemalloc.take_snapshot()
    finally:
        stopped.set()
        sampler.join(timeout=1)
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        tracemalloc.stop()
    retained_allocations = sum(
        max(0, statistic.count_diff) for statistic in after.compare_to(before, "lineno")
    )
    return {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": peak_rss[0],
        "peak_rss_delta_bytes": max(0, peak_rss[0] - initial_rss),
        "tracemalloc_current_bytes": current_traced,
        "tracemalloc_peak_bytes": traced_peak,
        "retained_allocation_count": retained_allocations,
        "serialized_bytes": output.stat().st_size,
    }


def _summarise(samples: list[dict[str, float | int]], rows: int) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")

    def median(name: str) -> float:
        return statistics.median(float(sample[name]) for sample in samples)

    serialized_sizes = {int(sample["serialized_bytes"]) for sample in samples}
    if len(serialized_sizes) != 1:
        raise RuntimeError(
            "serialized output size changed between deterministic repeats"
        )
    serialized_bytes = serialized_sizes.pop()
    wall_seconds = median("wall_seconds")
    return {
        "wall_seconds_median": wall_seconds,
        "cpu_seconds_median": median("cpu_seconds"),
        "peak_rss_bytes_max": max(int(sample["peak_rss_bytes"]) for sample in samples),
        "peak_rss_delta_bytes_max": max(
            int(sample["peak_rss_delta_bytes"]) for sample in samples
        ),
        "tracemalloc_peak_bytes_max": max(
            int(sample["tracemalloc_peak_bytes"]) for sample in samples
        ),
        "retained_allocation_count_median": median("retained_allocation_count"),
        "serialized_bytes": serialized_bytes,
        "bytes_per_row": serialized_bytes / rows,
        "throughput_rows_per_second_median": rows / wall_seconds,
        "samples": samples,
    }


def benchmark_serialization(rows: int = 20_000, repeats: int = 3) -> dict[str, object]:
    """Measure time, CPU, memory, allocations, and size for interchange formats."""
    if rows <= 0 or repeats <= 0:
        raise ValueError("rows and repeats must be positive")
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
        operations: dict[str, Callable[[], Path]] = {
            "jsonl": lambda: write_records(
                records, root / "records.jsonl", prefer_parquet=False
            ),
            "parquet": lambda: write_records(records, root / "records.parquet"),
            "ipc": lambda: write_ipc_records(records, root / "records.arrow"),
        }
        formats = {
            name: _summarise([_measure(operation) for _ in range(repeats)], rows)
            for name, operation in operations.items()
        }
    jsonl_wall = float(formats["jsonl"]["wall_seconds_median"])
    return {
        "schema_version": "2.0.0",
        "rows": rows,
        "repeats": repeats,
        "thread_controls": {
            name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES
        },
        "formats": formats,
        "ratios": {
            "parquet_write_vs_jsonl": float(formats["parquet"]["wall_seconds_median"])
            / jsonl_wall,
            "ipc_write_vs_jsonl": float(formats["ipc"]["wall_seconds_median"])
            / jsonl_wall,
        },
    }


def _number(
    mapping: Mapping[str, object], name: str, path: str
) -> tuple[float | None, str | None]:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{path}.{name}: missing or non-numeric"
    return float(value), None


def _format_failures(
    formats: Mapping[str, object], format_budgets: Mapping[str, object]
) -> list[str]:
    failures: list[str] = []
    for format_name, raw_limits in format_budgets.items():
        metrics = formats.get(format_name)
        if not isinstance(raw_limits, dict):
            failures.append(f"formats.{format_name}: invalid budget")
            continue
        if not isinstance(metrics, dict):
            failures.append(f"formats.{format_name}: missing benchmark result")
            continue
        typed_metrics = cast(dict[str, object], metrics)
        for budget_name, raw_limit in raw_limits.items():
            if (
                not isinstance(budget_name, str)
                or isinstance(raw_limit, bool)
                or not isinstance(raw_limit, (int, float))
            ):
                failures.append(f"formats.{format_name}.{budget_name}: invalid budget")
                continue
            direction, separator, metric_name = budget_name.partition("_")
            if separator != "_" or direction not in {"max", "min"}:
                failures.append(
                    f"formats.{format_name}.{budget_name}: invalid budget key"
                )
                continue
            actual, error = _number(
                typed_metrics, metric_name, f"formats.{format_name}"
            )
            if error:
                failures.append(error)
            elif actual is not None and (
                (direction == "max" and actual > float(raw_limit))
                or (direction == "min" and actual < float(raw_limit))
            ):
                failures.append(
                    f"formats.{format_name}.{metric_name}: {actual:.3f} "
                    f"violates {budget_name}={float(raw_limit):.3f}"
                )
    return failures


def _ratio_failures(
    ratios: Mapping[str, object], ratio_budgets: Mapping[str, object]
) -> list[str]:
    failures: list[str] = []
    for name, raw_limit in ratio_budgets.items():
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)):
            failures.append(f"maximum_ratios.{name}: invalid budget")
            continue
        actual, error = _number(ratios, name, "ratios")
        if error:
            failures.append(error)
        elif actual is not None and actual > float(raw_limit):
            failures.append(
                f"ratios.{name}: {actual:.3f} exceeds {float(raw_limit):.3f}"
            )
    return failures


def regression_failures(
    result: Mapping[str, object], budgets: Mapping[str, object]
) -> list[str]:
    """Fail closed when any committed performance envelope is missing or exceeded."""
    failures: list[str] = []
    formats = result.get("formats")
    format_budgets = budgets.get("formats")
    if not isinstance(formats, dict) or not isinstance(format_budgets, dict):
        return ["formats: missing benchmark results or budgets"]
    failures.extend(
        _format_failures(
            cast(dict[str, object], formats),
            cast(dict[str, object], format_budgets),
        )
    )
    ratios = result.get("ratios")
    ratio_budgets = budgets.get("maximum_ratios")
    if not isinstance(ratios, dict) or not isinstance(ratio_budgets, dict):
        failures.append("maximum_ratios: missing benchmark results or budgets")
    else:
        failures.extend(
            _ratio_failures(
                cast(dict[str, object], ratios),
                cast(dict[str, object], ratio_budgets),
            )
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--budgets", type=Path, default=Path("benchmarks/serialization_budgets.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    thread_failures = blas_thread_failures(os.environ)
    if thread_failures:
        print("BLAS thread controls must equal 1: " + ", ".join(thread_failures))
        return 2
    result = benchmark_serialization(args.rows, args.repeats)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    raw_budgets: Any = json.loads(args.budgets.read_text(encoding="utf-8"))
    if not isinstance(raw_budgets, dict):
        print("budgets: expected a JSON object")
        return 2
    failures = regression_failures(result, raw_budgets)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
