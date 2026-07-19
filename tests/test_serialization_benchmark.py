from __future__ import annotations

from scripts.serialization_benchmark import regression_failures


def test_regression_budget_comparison() -> None:
    result = {"ratios": {"parquet": 1.25, "ipc": 2.5}}
    assert regression_failures(result, {"parquet": 2.0, "ipc": 2.0}) == [
        "ipc: 2.500 exceeds 2.000"
    ]
