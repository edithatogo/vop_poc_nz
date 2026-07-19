from __future__ import annotations

from typing import cast

import pytest

from scripts.serialization_benchmark import (
    BLAS_THREAD_VARIABLES,
    benchmark_serialization,
    blas_thread_failures,
    regression_failures,
)


def _result(metric: float = 2.0) -> dict[str, object]:
    return {
        "formats": {"ipc": {"wall": metric, "throughput": metric}},
        "ratios": {"ipc": metric},
    }


def _budgets() -> dict[str, object]:
    return {
        "formats": {"ipc": {"max_wall": 2.5, "min_throughput": 1.5}},
        "maximum_ratios": {"ipc": 2.5},
    }


def test_regression_budget_comparison_passes_complete_envelope() -> None:
    assert regression_failures(_result(), _budgets()) == []


@pytest.mark.parametrize(
    ("result", "fragment"),
    [
        (_result(3.0), "violates max_wall"),
        (_result(1.0), "violates min_throughput"),
        ({"formats": {"ipc": {}}, "ratios": {"ipc": 2.0}}, "missing or non-numeric"),
        (
            {"formats": {"ipc": {"wall": 2.0, "throughput": 2.0}}, "ratios": {}},
            "ratios.ipc",
        ),
    ],
)
def test_regression_budget_comparison_fails_closed(
    result: dict[str, object], fragment: str
) -> None:
    assert any(
        fragment in failure for failure in regression_failures(result, _budgets())
    )


def test_benchmark_reports_multidimensional_deterministic_envelope() -> None:
    result = benchmark_serialization(rows=25, repeats=2)
    formats = cast(dict[str, dict[str, object]], result["formats"])
    assert set(formats) == {"jsonl", "parquet", "ipc"}
    for metrics in formats.values():
        assert len(cast(list[object], metrics["samples"])) == 2
        numeric = {
            name: value
            for name, value in metrics.items()
            if name != "samples" and isinstance(value, (int, float))
        }
        assert numeric["wall_seconds_median"] > 0
        assert numeric["cpu_seconds_median"] >= 0
        assert numeric["peak_rss_bytes_max"] > 0
        assert numeric["peak_rss_delta_bytes_max"] >= 0
        assert numeric["tracemalloc_peak_bytes_max"] > 0
        assert numeric["positive_snapshot_allocation_count_median"] >= 0
        assert numeric["serialized_bytes"] > 0
        assert numeric["bytes_per_row"] > 0
        assert numeric["throughput_rows_per_second_median"] > 0


def test_blas_thread_controls_require_every_backend_to_equal_one() -> None:
    environment = dict.fromkeys(BLAS_THREAD_VARIABLES, "1")
    assert blas_thread_failures(environment) == []
    environment["MKL_NUM_THREADS"] = "2"
    assert blas_thread_failures(environment) == ["MKL_NUM_THREADS"]


@pytest.mark.parametrize(("rows", "repeats"), [(0, 1), (1, 0), (-1, 1)])
def test_benchmark_rejects_non_positive_dimensions(rows: int, repeats: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        benchmark_serialization(rows=rows, repeats=repeats)
