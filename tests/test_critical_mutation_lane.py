"""Contract tests for the strict critical mutation lane."""

from __future__ import annotations

import pytest

from scripts.run_critical_mutation_lane import mutation_report


def _stats(**changes: int) -> dict[str, object]:
    values: dict[str, object] = {
        "killed": 90,
        "survived": 5,
        "no_tests": 2,
        "suspicious": 1,
        "timeout": 1,
        "segfault": 1,
        "skipped": 3,
        "check_was_interrupted_by_user": 0,
        "total": 103,
    }
    values.update(changes)
    return values


def test_report_counts_all_non_skipped_outcomes() -> None:
    report = mutation_report(_stats(), threshold=90.0)
    assert report["eligible"] == 100
    assert report["score_percent"] == 90.0
    assert report["passed"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"killed": 89, "survived": 6},
        {"check_was_interrupted_by_user": 1, "total": 104},
        {
            "total": 3,
            "skipped": 3,
            "killed": 0,
            "survived": 0,
            "no_tests": 0,
            "suspicious": 0,
            "timeout": 0,
            "segfault": 0,
        },
    ],
)
def test_report_fails_below_threshold_interruption_or_empty(
    changes: dict[str, int],
) -> None:
    assert mutation_report(_stats(**changes), threshold=90.0)["passed"] is False


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_report_rejects_invalid_counts(value: object) -> None:
    stats = _stats()
    stats["killed"] = value
    with pytest.raises(ValueError, match="non-negative integer"):
        mutation_report(stats, threshold=90.0)


def test_report_rejects_total_smaller_than_accounted_statuses() -> None:
    with pytest.raises(ValueError, match="smaller"):
        mutation_report(_stats(total=99), threshold=90.0)


@pytest.mark.parametrize("threshold", [0.0, -1.0, 100.1])
def test_report_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        mutation_report(_stats(), threshold=threshold)
