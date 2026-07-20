#!/usr/bin/env python3
"""Profile a bounded, privacy-safe governance sync-planning workload."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from vop_poc_nz.concerns import GovernanceLedger, build_github_sync_payloads
from vop_poc_nz.github_sync_planner import GitHubIssueSnapshot, plan_github_sync


def main() -> int:
    """Run bounded planner iterations and write aggregate evidence only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("governance/registry.json"))
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/governance-profile.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10_000:
        raise ValueError("iterations must be between 1 and 10000")
    ledger = GovernanceLedger.model_validate_json(
        args.ledger.read_text(encoding="utf-8")
    )
    payload = build_github_sync_payloads(ledger)[0]
    base = GitHubIssueSnapshot(
        github_repository=payload.github_repository,
        issue_number=payload.issue_number,
        state=payload.desired_state,
        title=payload.title,
        body=payload.body,
        labels=payload.labels,
        project_number=payload.project_number,
        project_fields=payload.project_fields,
    )
    started = perf_counter()
    plan = None
    for _ in range(args.iterations):
        plan = plan_github_sync(base=base, local=payload, remote=base)
    elapsed = perf_counter() - started
    assert plan is not None
    report = {
        "schema_version": "1.0.0",
        "workload": "governance_three_way_sync_planner",
        "iterations": args.iterations,
        "elapsed_seconds": round(elapsed, 9),
        "plans_per_second": round(args.iterations / elapsed, 3),
        "outcome": plan.outcome,
        "projection_digest": sha256(plan.local_digest.encode("ascii")).hexdigest(),
        "network_mutation": False,
        "private_content_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
