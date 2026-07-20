from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from scripts.check_c15_mutation_cohort import (
    cohort_identity,
    evaluate_cohort,
    mutation_universe,
    validate_runtime_version,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads(
    (ROOT / ".github/mutation-baselines/vop-c15-cohort.json").read_text(
        encoding="utf-8"
    )
)
ANCHOR = "a" * 64


def _baseline_statuses() -> dict[str, str]:
    stats = cast("dict[str, int]", BASELINE["stats"])
    ids = cast("list[str]", BASELINE["universe"]["ids"])
    status_counts = (
        ("killed", stats["killed"]),
        ("survived", stats["survived"]),
        ("no tests", stats["no_tests"]),
        ("skipped", stats["skipped"]),
        ("suspicious", stats["suspicious"]),
        ("timeout", stats["timeout"]),
        (
            "check was interrupted by user",
            stats["check_was_interrupted_by_user"],
        ),
        ("segfault", stats["segfault"]),
    )
    statuses = [
        status for status, count in status_counts for _ in range(count)
    ]
    statuses.extend("caught by type check" for _ in range(stats["total"] - len(statuses)))
    assert ids and len(ids) == len(statuses)
    return dict(zip(ids, statuses, strict=True))


def _universe(*, replacement: bool = False) -> dict[str, object]:
    statuses = _baseline_statuses()
    if replacement:
        replaced_id = next(reversed(statuses))
        statuses["vop_poc_nz.example__mutmut_replacement"] = statuses.pop(replaced_id)
    return mutation_universe(
        "\n".join(f"{mutant}: {status}" for mutant, status in statuses.items())
    )


def _reviewed(identity: dict[str, object]) -> dict[str, object]:
    baseline = deepcopy(BASELINE)
    ids = _universe()["ids"]
    stats = cast("dict[str, int]", baseline["stats"])
    policy = cast("dict[str, object]", baseline["policy"])
    eligible = stats["total"] - stats["skipped"]
    debt = eligible - stats["killed"]
    baseline["cohort"] = identity
    baseline["universe"] = {
        "ids": ids,
        "sha256": sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "promotion_state": "captured",
    }
    # Exercise an attainable exact boundary in the synthetic reviewed fixture.
    # The promoted file intentionally retains its independently reviewed,
    # display-rounded policy values and is not rewritten by unit tests.
    policy["minimum_score_percent"] = 100.0 * stats["killed"] / eligible
    policy["maximum_debt_density"] = debt / eligible
    return baseline


def _evaluate(
    identity: dict[str, object],
    *,
    stats: dict[str, object] | None = None,
    universe: dict[str, object] | None = None,
    baseline: dict[str, object] | None = None,
    reviewed: str = ANCHOR,
) -> dict[str, object]:
    selected_baseline = baseline or _reviewed(identity)
    policy = cast("dict[str, object]", selected_baseline["policy"])
    return evaluate_cohort(
        stats or BASELINE["stats"],
        selected_baseline,
        identity,
        universe or _universe(),
        float(policy["minimum_score_percent"]),
        baseline_sha256=ANCHOR,
        reviewed_baseline_sha256=reviewed,
    )


def _universe_with_final_status(status: str) -> dict[str, object]:
    statuses = _baseline_statuses()
    replaced_id = next(
        mutant_id
        for mutant_id, current_status in reversed(statuses.items())
        if current_status == "survived"
    )
    statuses[replaced_id] = status
    return mutation_universe(
        "\n".join(f"{mutant}: {current}" for mutant, current in statuses.items())
    )


def test_cohort_binds_tool_lock_config_source_universe_and_debt() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    report = _evaluate(identity)
    assert report["passed"] is True
    assert report["debt"]["absolute"] == BASELINE["policy"]["maximum_absolute_debt"]
    assert report["universe"]["matches"] is True
    validate_runtime_version(identity, "3.6.0")


def test_external_anchor_and_drift_fail_closed() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    assert _evaluate(identity, reviewed="")["passed"] is False
    assert "human_approved" not in BASELINE["promotion_provenance"]
    drifted = deepcopy(identity)
    drifted["lock_sha256"] = "0" * 64
    assert _evaluate(drifted, baseline=_reviewed(identity))["passed"] is False
    report = _evaluate(identity, universe=_universe(replacement=True))
    assert report["passed"] is False
    assert len(report["universe"]["added_ids"]) == 1


def test_runtime_and_universe_parser_reject_invalid_inputs() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    try:
        validate_runtime_version(identity, "3.5.0")
    except ValueError:
        pass
    else:
        raise AssertionError("runtime drift was accepted")
    for value in (
        "pkg.fn__mutmut_1: mystery",
        "pkg.fn__mutmut_1: killed\npkg.fn__mutmut_1: survived",
    ):
        try:
            mutation_universe(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid universe was accepted")


def test_status_partition_rejects_not_checked_and_accounts_for_type_checks() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    stats = deepcopy(BASELINE["stats"])
    stats["survived"] = cast("int", stats["survived"]) - 1
    with pytest.raises(ValueError, match="not checked"):
        _evaluate(
            identity,
            stats=stats,
            universe=_universe_with_final_status("not checked"),
        )

    report = _evaluate(
        identity,
        stats=stats,
        universe=_universe_with_final_status("caught by type check"),
    )
    assert report["passed"] is True
    expected_counts = {
        "caught by type check": 1,
        "check was interrupted by user": cast(
            "int", stats["check_was_interrupted_by_user"]
        ),
        "killed": cast("int", stats["killed"]),
        "no tests": cast("int", stats["no_tests"]),
        "not checked": 0,
        "segfault": cast("int", stats["segfault"]),
        "skipped": cast("int", stats["skipped"]),
        "suspicious": cast("int", stats["suspicious"]),
        "survived": cast("int", stats["survived"]),
        "timeout": cast("int", stats["timeout"]),
    }
    assert report["status_partition"] == {
        "complete": True,
        "counts": expected_counts,
        "total": cast("int", stats["total"]),
    }


def test_status_partition_rejects_incomplete_statistics() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    with pytest.raises(ValueError, match="statuses do not match statistics"):
        _evaluate(
            identity,
            universe=_universe_with_final_status("caught by type check"),
        )
    incomplete = deepcopy(_universe())
    incomplete_statuses = cast("dict[str, str]", incomplete["statuses"])
    incomplete_ids = cast("list[str]", incomplete["ids"])
    incomplete_statuses.pop(incomplete_ids[0])
    with pytest.raises(ValueError, match="partition is incomplete"):
        _evaluate(identity, universe=incomplete)
    unknown = deepcopy(_universe())
    unknown_statuses = cast("dict[str, str]", unknown["statuses"])
    unknown_ids = cast("list[str]", unknown["ids"])
    unknown_statuses[unknown_ids[0]] = "unknown"
    with pytest.raises(ValueError, match="partition is incomplete"):
        _evaluate(identity, universe=unknown)
