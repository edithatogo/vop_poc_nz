"""Tests for truthful frozen, supported-latest, absolute, and preview policy."""

from pathlib import Path
from typing import cast

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from scripts.dependency_frontier import dependency_report, release_frontier


def _payload() -> dict[str, object]:
    def files(requires_python: str = ">=3.14") -> list[dict[str, object]]:
        return [{"requires_python": requires_python, "yanked": False}]

    return {
        "releases": {
            "1.0.0": files(),
            "1.5.0": files(),
            "1.6.0rc1": files(),
            "2.0.0": files(),
            "3.0.0": files(">=3.15"),
            "9.0.0": [{"requires_python": ">=3.14", "yanked": True}],
        }
    }


def test_release_frontier_separates_supported_absolute_and_prerelease() -> None:
    frontier = release_frontier(
        _payload(), Requirement("example>=1,<2"), python_version=Version("3.14")
    )
    assert frontier == {
        "supported_latest_stable": "1.5.0",
        "absolute_latest_stable": "2.0.0",
        "latest_prerelease": "1.6.0rc1",
    }


def test_stable_lane_validates_frozen_lock_without_live_frontier() -> None:
    report = dependency_report(
        [("core", Requirement("example>=1,<2"))],
        {"example": Version("1.0.0")},
        lane="stable",
        python_version=Version("3.14"),
        lock_digest="0" * 64,
    )
    assert report["passed"] is True
    assert report["supported_latest_passed"] is True
    dependencies = cast(list[dict[str, object]], report["dependencies"])
    assert "supported_latest_stable" not in dependencies[0]


def test_build_frontend_requirements_are_reported_but_not_uv_lock_managed() -> None:
    report = dependency_report(
        [("build", Requirement("builder>=1"))],
        {},
        lane="stable",
        python_version=Version("3.14"),
        lock_digest="0" * 64,
    )
    row = cast(list[dict[str, object]], report["dependencies"])[0]
    assert report["passed"] is True
    assert row["stable_frozen_valid"] is None
    assert row["resolution_policy"] == "build_frontend_isolated"


def test_frontier_lane_requires_latest_supported_but_not_absolute_or_preview() -> None:
    requirements = [("core", Requirement("example>=1,<2"))]
    pypi = {"example": _payload()}
    behind = dependency_report(
        requirements,
        {"example": Version("1.0.0")},
        lane="frontier",
        python_version=Version("3.14"),
        pypi=pypi,
        lock_digest="0" * 64,
    )
    current = dependency_report(
        requirements,
        {"example": Version("1.5.0")},
        lane="frontier",
        python_version=Version("3.14"),
        pypi=pypi,
        lock_digest="1" * 64,
    )
    assert behind["passed"] is False
    assert current["passed"] is True
    row = cast(list[dict[str, object]], current["dependencies"])[0]
    assert row["absolute_stable_blocked_by_policy"] is True
    assert row["prerelease_policy"] == "report_only"


def test_workflow_keeps_frozen_and_ephemeral_frontier_lanes_separate() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/dependency-frontier.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    stable = str(jobs["stable-frozen"])
    frontier = str(jobs["supported-latest"])
    assert "uv lock --check" in stable
    assert "--lane stable" in stable
    assert "uv lock --upgrade" not in stable
    assert "uv lock --upgrade" in frontier
    assert "--lane frontier" in frontier
    assert "uv.lock" in frontier
