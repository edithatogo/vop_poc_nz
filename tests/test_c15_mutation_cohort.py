from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

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


def _universe(*, replacement: bool = False, killed: int = 370) -> dict[str, object]:
    ids = [f"vop_poc_nz.example__mutmut_{number}" for number in range(827)]
    if replacement:
        ids[-1] = "vop_poc_nz.example__mutmut_replacement"
    return mutation_universe(
        "\n".join(
            f"{mutant}: {'killed' if index < killed else 'survived'}"
            for index, mutant in enumerate(ids)
        )
    )


def _reviewed(identity: dict[str, object]) -> dict[str, object]:
    baseline = deepcopy(BASELINE)
    ids = _universe()["ids"]
    baseline["cohort"] = identity
    baseline["universe"] = {
        "ids": ids,
        "sha256": sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "promotion_state": "captured",
    }
    return baseline


def _evaluate(
    identity: dict[str, object],
    *,
    universe: dict[str, object] | None = None,
    baseline: dict[str, object] | None = None,
    reviewed: str = ANCHOR,
) -> dict[str, object]:
    return evaluate_cohort(
        BASELINE["stats"],
        baseline or _reviewed(identity),
        identity,
        universe or _universe(),
        44.0,
        baseline_sha256=ANCHOR,
        reviewed_baseline_sha256=reviewed,
    )


def test_cohort_binds_tool_lock_config_source_universe_and_debt() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    report = _evaluate(identity)
    assert report["passed"] is True
    assert report["debt"]["absolute"] == 457
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
