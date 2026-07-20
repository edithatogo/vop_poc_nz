#!/usr/bin/env python3
"""Write a local, deterministic GitHub governance sync plan without mutation."""

from __future__ import annotations

import argparse
from pathlib import Path

from vop_poc_nz.concerns import GovernanceLedger, build_github_sync_payloads
from vop_poc_nz.github_sync_planner import (
    issue_snapshot_from_json,
    plan_github_sync,
    sync_plan_json,
)


def main() -> int:
    """Plan one issue projection from ledger, base, and remote snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("governance/registry.json"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--close-approved", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".conductor/local/github_sync_plan.json"),
    )
    args = parser.parse_args()

    ledger = GovernanceLedger.model_validate_json(
        args.ledger.read_text(encoding="utf-8")
    )
    marker = f"vop-voiage-governance-id:{args.record_id}"
    payloads = {
        payload.stable_marker: payload for payload in build_github_sync_payloads(ledger)
    }
    if marker not in payloads:
        raise ValueError(f"no public issue projection for {args.record_id}")

    plan = plan_github_sync(
        base=issue_snapshot_from_json(args.base.read_text(encoding="utf-8")),
        local=payloads[marker],
        remote=issue_snapshot_from_json(args.remote.read_text(encoding="utf-8")),
        close_approved=args.close_approved,
    )
    repository = Path.cwd().resolve()
    local_root = (repository / ".conductor/local").resolve()
    output = (repository / args.output).resolve()
    if not output.is_relative_to(local_root):
        raise ValueError("sync plans may only be written under .conductor/local")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(sync_plan_json(plan), encoding="utf-8", newline="\n")
    print(output)
    return 2 if plan.outcome == "conflict" else 0


if __name__ == "__main__":
    raise SystemExit(main())
