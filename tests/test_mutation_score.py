"""Tests for the bounded Mutmut score gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vop_poc_nz.mutation_policy import (
    mutation_score_from_mapping,
    mutation_score_from_meta,
    mutation_target_report,
    validate_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def _stats(**changes: int) -> dict[str, int]:
    values = {
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


def test_score_counts_every_non_skipped_unresolved_mutant() -> None:
    score = mutation_score_from_mapping(_stats())
    report = score.report(90.0)
    assert report["eligible"] == 100
    assert report["score_percent"] == 90.0
    assert report["passed"] is True
    assert report["survived"] == 5
    assert report["no_tests"] == 2


def test_unreported_mutmut_statuses_cannot_inflate_score() -> None:
    report = mutation_score_from_mapping(_stats(total=104)).report(90.0)
    assert report["eligible"] == 101
    assert report["passed"] is False


def test_exact_baseline_comparison_rejects_a_rounded_regression() -> None:
    baseline = mutation_score_from_mapping(
        _stats(
            killed=90,
            survived=10,
            no_tests=0,
            suspicious=0,
            timeout=0,
            segfault=0,
            total=103,
        )
    )
    regressed = mutation_score_from_mapping(
        _stats(
            killed=899,
            survived=101,
            no_tests=0,
            suspicious=0,
            timeout=0,
            segfault=0,
            skipped=0,
            total=1000,
        )
    )
    report = regressed.report(89.0, baseline=baseline)
    assert report["score_percent"] == 89.9
    assert report["non_decreasing"] is False
    assert report["passed"] is False


def test_target_ratchet_allows_only_killed_universe_growth() -> None:
    improved = mutation_score_from_mapping(
        _stats(
            killed=91,
            survived=5,
            no_tests=2,
            suspicious=1,
            timeout=1,
            segfault=1,
            total=104,
        )
    )
    report = mutation_target_report(improved, baseline_killed=90, baseline_eligible=100)
    assert report["universe_delta"] == 1
    assert report["unresolved"] == report["baseline_unresolved"] == 10
    assert report["score_non_decreasing"] is True
    assert report["debt_non_increasing"] is True
    assert report["passed"] is True


def test_target_ratchet_rejects_new_unresolved_mutation_debt() -> None:
    regressed = mutation_score_from_mapping(_stats(total=104))
    report = mutation_target_report(
        regressed, baseline_killed=90, baseline_eligible=100
    )
    assert report["universe_delta"] == 1
    assert report["unresolved"] == 11
    assert report["score_non_decreasing"] is False
    assert report["debt_non_increasing"] is False
    assert report["passed"] is False


def test_mutmut_meta_preserves_unreported_statuses_as_debt(tmp_path: Path) -> None:
    meta = tmp_path / "target.py.meta"
    meta.write_text(
        json.dumps(
            {
                "exit_code_by_key": {
                    "pkg.x__mutmut_1": 1,
                    "pkg.x__mutmut_2": 0,
                    "pkg.x__mutmut_3": 33,
                    "pkg.x__mutmut_4": None,
                    "pkg.x__mutmut_5": 37,
                    "pkg.x__mutmut_6": 34,
                }
            }
        ),
        encoding="utf-8",
    )
    score = mutation_score_from_meta(meta)
    assert score.total == 6
    assert score.skipped == 1
    assert score.eligible == 5
    assert score.killed == 1
    assert score.survived == score.no_tests == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"killed": 89, "survived": 6},
        {"check_was_interrupted_by_user": 1, "total": 104},
        {
            "killed": 0,
            "survived": 0,
            "no_tests": 0,
            "suspicious": 0,
            "timeout": 0,
            "segfault": 0,
            "total": 3,
        },
    ],
)
def test_score_fails_below_threshold_interrupted_or_empty(
    changes: dict[str, int],
) -> None:
    assert (
        mutation_score_from_mapping(_stats(**changes)).report(90.0)["passed"] is False
    )


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_invalid_counts_fail_closed(value: object) -> None:
    raw: dict[str, object] = dict(_stats())
    raw["killed"] = value
    with pytest.raises(ValueError, match="non-negative integer"):
        mutation_score_from_mapping(raw)


def test_inconsistent_total_and_invalid_threshold_fail_closed() -> None:
    with pytest.raises(ValueError, match="smaller"):
        mutation_score_from_mapping(_stats(total=99))
    for threshold in (0.0, -1.0, 100.1):
        with pytest.raises(ValueError, match="threshold"):
            validate_threshold(threshold)


def test_cli_reports_counts_and_returns_pass_or_fail(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    report = tmp_path / "report.json"
    stats.write_text(json.dumps(_stats()), encoding="utf-8")
    passed = subprocess.run(
        [
            sys.executable,
            "scripts/check_mutation_score.py",
            "--stats",
            str(stats),
            "--output",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True
    stats.write_text(json.dumps(_stats(killed=89, survived=6)), encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, "scripts/check_mutation_score.py", "--stats", str(stats)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["passed"] is False


def test_cli_rejects_score_regression_even_above_floor(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    baseline = tmp_path / "baseline.json"
    stats.write_text(
        json.dumps(
            _stats(
                killed=899,
                survived=101,
                no_tests=0,
                suspicious=0,
                timeout=0,
                segfault=0,
                skipped=0,
                total=1000,
            )
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            _stats(
                killed=90,
                survived=10,
                no_tests=0,
                suspicious=0,
                timeout=0,
                segfault=0,
                total=103,
            )
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_mutation_score.py",
            "--stats",
            str(stats),
            "--threshold",
            "89",
            "--baseline-stats",
            str(baseline),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["score_percent"] == 89.9
    assert report["non_decreasing"] is False


def test_target_cli_passes_baseline_and_rejects_new_debt(tmp_path: Path) -> None:
    cache = tmp_path / "mutants"
    meta = cache / "src/target.py.meta"
    meta.parent.mkdir(parents=True)
    statuses = {
        "pkg.x__mutmut_1": 1,
        "pkg.x__mutmut_2": 1,
        "pkg.x__mutmut_3": 0,
    }
    meta.write_text(json.dumps({"exit_code_by_key": statuses}), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "targets": {
                    "src/target.py": {
                        "cache_metadata": "src/target.py.meta",
                        "killed": 2,
                        "eligible": 3,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "scripts/check_mutation_targets.py",
        "--baseline",
        str(baseline),
        "--cache-root",
        str(cache),
    ]
    passed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["passed"] is True

    statuses["pkg.x__mutmut_4"] = 0
    meta.write_text(json.dumps({"exit_code_by_key": statuses}), encoding="utf-8")
    failed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["targets"]["src/target.py"]["unresolved"] == 2
