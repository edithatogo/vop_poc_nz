from __future__ import annotations

from typing import cast

import pytest

from scripts.check_scalene_profile import scalene_profile_report


def _profile(elapsed: float = 1.0) -> dict[str, object]:
    return {
        "program": "/workspace/scripts/profile_workload.py",
        "elapsed_time_sec": elapsed,
        "files": {"/workspace/scripts/profile_workload.py": {"lines": []}},
    }


def _baseline() -> dict[str, object]:
    return {
        "required_scalene_version": "2.3.0",
        "workload": "profile_workload.py",
        "reference_elapsed_seconds": 1.0,
        "maximum_elapsed_seconds": 5.0,
        "maximum_regression_factor": 3.0,
        "reference_environment": "synthetic-test",
    }


def test_profile_report_passes_pinned_bounded_observation() -> None:
    report = scalene_profile_report(_profile(), _baseline(), scalene_version="2.3.0")
    assert report["passed"] is True
    assert report["profiled_file_count"] == 1
    assert report["elapsed_regression_factor"] == 1.0


@pytest.mark.parametrize(
    ("profile", "version", "fragment"),
    [
        (_profile(4.0), "2.3.0", "elapsed_regression_factor"),
        (_profile(6.0), "2.3.0", "elapsed_time_sec"),
        (_profile(), "2.4.0", "scalene_version"),
    ],
)
def test_profile_report_fails_regressions_or_version_drift(
    profile: dict[str, object], version: str, fragment: str
) -> None:
    report = scalene_profile_report(profile, _baseline(), scalene_version=version)
    assert report["passed"] is False
    assert any(fragment in failure for failure in cast(list[str], report["failures"]))


@pytest.mark.parametrize(
    "profile",
    [
        {"program": "profile_workload.py", "elapsed_time_sec": 1.0, "files": {}},
        {"program": "other.py", "elapsed_time_sec": 1.0, "files": {"x": {}}},
        {"program": "profile_workload.py", "elapsed_time_sec": "1", "files": {"x": {}}},
    ],
)
def test_profile_report_fails_closed_on_incomplete_profiles(
    profile: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        scalene_profile_report(profile, _baseline(), scalene_version="2.3.0")
