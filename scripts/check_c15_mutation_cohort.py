#!/usr/bin/env python3
"""Bind VOP mutation debt and density to an externally reviewed exact cohort."""

# pyright: reportAny=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import tomllib
from hashlib import sha256
from pathlib import Path
from typing import cast

from vop_poc_nz.mutation_policy import (
    mutation_score_from_mapping,
    validate_threshold,
)

_STATUSES = frozenset(
    {
        "killed",
        "survived",
        "no tests",
        "suspicious",
        "timeout",
        "segfault",
        "skipped",
        "check was interrupted by user",
        "not checked",
        "caught by type check",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", value)


def cohort_identity(repo: Path, config_path: Path) -> dict[str, object]:
    """Derive source/config/tool/lock identity without importing Mutmut."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    mutmut = cast(
        "dict[str, object]", cast("dict[str, object]", config["tool"])["mutmut"]
    )
    targets = cast("list[str]", mutmut["only_mutate"])
    sources: list[dict[str, object]] = []
    logical_lines = 0
    for relative in sorted(targets):
        data = (repo / relative).read_bytes()
        logical_lines += sum(
            bool(line.strip()) and not line.lstrip().startswith("#")
            for line in data.decode("utf-8").splitlines()
        )
        sources.append({"path": relative, "sha256": sha256(data).hexdigest()})
    lock_data = (repo / "uv.lock").read_bytes()
    packages = cast(
        "list[dict[str, object]]", tomllib.loads(lock_data.decode("utf-8"))["package"]
    )
    versions = [
        item.get("version") for item in packages if item.get("name") == "mutmut"
    ]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise ValueError("uv.lock must contain exactly one Mutmut version")
    cohort = {
        "tool": "mutmut",
        "tool_version": versions[0],
        "lock_sha256": sha256(lock_data).hexdigest(),
        "configuration_sha256": sha256(_canonical(mutmut)).hexdigest(),
        "sources": sources,
    }
    return {
        **cohort,
        "source_logical_lines": logical_lines,
        "cohort_sha256": sha256(_canonical(cohort)).hexdigest(),
    }


def validate_runtime_version(identity: dict[str, object], runtime_version: str) -> None:
    """Reject an executing Mutmut version different from the locked cohort."""
    if runtime_version != identity.get("tool_version"):
        raise ValueError("installed Mutmut version does not match the locked cohort")


def mutation_universe(text: str) -> dict[str, object]:
    """Parse the complete stable mutant ID/status inventory."""
    statuses: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        mutant_id, separator, status = line.rpartition(": ")
        if not separator or not mutant_id or status not in _STATUSES:
            raise ValueError(f"invalid mutation-universe row: {line}")
        if mutant_id in statuses:
            raise ValueError(f"duplicate mutation identity: {mutant_id}")
        statuses[mutant_id] = status
    ids = sorted(statuses)
    return {
        "ids": ids,
        "sha256": sha256(_canonical(ids)).hexdigest(),
        "statuses": {mutant_id: statuses[mutant_id] for mutant_id in ids},
    }


def evaluate_cohort(
    stats: dict[str, object],
    baseline: dict[str, object],
    identity: dict[str, object],
    universe: dict[str, object],
    threshold: float,
    *,
    baseline_sha256: str,
    reviewed_baseline_sha256: str,
) -> dict[str, object]:
    """Apply score, debt, density, universe, provenance, and external-anchor gates."""
    if baseline.get("schema_version") != "1.0.0":
        raise ValueError("unsupported mutation cohort baseline schema")
    provenance = cast("dict[str, object]", baseline["promotion_provenance"])
    commit = provenance.get("commit")
    provenance_valid = (
        provenance.get("review_state") == "requires_external_anchor"
        and isinstance(provenance.get("run_id"), int)
        and not isinstance(provenance.get("run_id"), bool)
        and isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
        and isinstance(provenance.get("evidence_url"), str)
        and cast("str", provenance["evidence_url"]).startswith("https://github.com/")
    )
    external_anchor_valid = (
        len(reviewed_baseline_sha256) == 64
        and all(
            character in "0123456789abcdef" for character in reviewed_baseline_sha256
        )
        and reviewed_baseline_sha256 == baseline_sha256
    )
    expected_identity = cast("dict[str, object]", baseline["cohort"])
    current_ids = cast("list[str]", universe["ids"])
    score = mutation_score_from_mapping(stats)
    if len(current_ids) != score.total:
        raise ValueError(
            "mutation universe cardinality does not match statistics total"
        )
    baseline_universe = cast("dict[str, object]", baseline["universe"])
    baseline_ids = cast("list[str]", baseline_universe["ids"])
    if (
        baseline_universe.get("sha256")
        != sha256(_canonical(sorted(baseline_ids))).hexdigest()
    ):
        raise ValueError("baseline mutation universe digest mismatch")
    statuses = cast("dict[str, str]", universe["statuses"])
    expected_counts = {
        "killed": score.killed,
        "survived": score.survived,
        "no tests": score.no_tests,
        "suspicious": score.suspicious,
        "timeout": score.timeout,
        "segfault": score.segfault,
        "skipped": score.skipped,
        "check was interrupted by user": score.interrupted,
    }
    if any(
        sum(value == status for value in statuses.values()) != count
        for status, count in expected_counts.items()
    ):
        raise ValueError("mutation universe statuses do not match statistics")
    omitted = score.total - sum(expected_counts.values())
    observed_omitted = sum(
        status in {"not checked", "caught by type check"}
        for status in statuses.values()
    )
    if observed_omitted != omitted:
        raise ValueError("mutation universe omitted statuses do not match statistics")
    added_ids = sorted(set(current_ids) - set(baseline_ids))
    removed_ids = sorted(set(baseline_ids) - set(current_ids))
    baseline_score = mutation_score_from_mapping(
        cast("dict[str, object]", baseline["stats"])
    )
    score_report = score.report(validate_threshold(threshold), baseline=baseline_score)
    policy = cast("dict[str, object]", baseline["policy"])
    debt = score.eligible - score.killed
    density = debt / score.eligible if score.eligible else 1.0
    logical_lines = int(cast("int", identity["source_logical_lines"]))
    passed = (
        identity == expected_identity
        and provenance_valid
        and external_anchor_valid
        and not added_ids
        and not removed_ids
        and threshold >= float(cast("int | float", policy["minimum_score_percent"]))
        and bool(score_report["passed"])
        and debt <= int(cast("int", policy["maximum_absolute_debt"]))
        and density <= float(cast("int | float", policy["maximum_debt_density"]))
    )
    return {
        "schema_version": "1.0.0",
        "cohort": identity,
        "identity_matches": identity == expected_identity,
        "promotion_provenance": provenance,
        "promotion_provenance_valid": provenance_valid,
        "baseline_sha256": baseline_sha256,
        "reviewed_baseline_sha256": reviewed_baseline_sha256,
        "external_review_anchor_valid": external_anchor_valid,
        "universe": {
            **universe,
            "baseline_sha256": baseline_universe.get("sha256"),
            "added_ids": added_ids,
            "removed_ids": removed_ids,
            "matches": not added_ids and not removed_ids,
        },
        "score": score_report,
        "debt": {
            "absolute": debt,
            "maximum_absolute": policy["maximum_absolute_debt"],
            "density": round(density, 6),
            "maximum_density": policy["maximum_debt_density"],
            "mutants_per_kloc": round(score.eligible * 1000 / logical_lines, 3),
        },
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats", type=Path, default=Path("mutants/mutmut-cicd-stats.json")
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--reviewed-baseline-sha256", required=True)
    parser.add_argument("--config", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--threshold", type=float, default=44.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    baseline_bytes = args.baseline.read_bytes()
    identity = cohort_identity(repo, repo / args.config)
    from importlib.metadata import version

    validate_runtime_version(identity, version("mutmut"))
    report = evaluate_cohort(
        _object(args.stats),
        _object(args.baseline),
        identity,
        mutation_universe(args.universe.read_text(encoding="utf-8")),
        args.threshold,
        baseline_sha256=sha256(baseline_bytes).hexdigest(),
        reviewed_baseline_sha256=args.reviewed_baseline_sha256,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
