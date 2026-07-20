#!/usr/bin/env python3
"""Emit a read-only GitHub governance reconciliation artifact."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from vop_poc_nz.concerns import GovernanceLedger, build_github_sync_payloads
from vop_poc_nz.github_drift_auditor import (
    audit_governance_drift,
    fetch_issue,
    fetch_project_check,
    governance_audit_exit_code,
    governance_baseline_from_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("governance/registry.json"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".conductor/local/github_governance_drift.json"),
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
    local = payloads[marker]
    if local.issue_number is None:
        raise ValueError(f"governance record {args.record_id} has no GitHub issue")
    base, baseline_provenance = governance_baseline_from_json(
        args.base.read_text(encoding="utf-8")
    )

    issue = fetch_issue(
        local.github_repository,
        local.issue_number,
        token=os.getenv("GITHUB_TOKEN"),
    )
    project = fetch_project_check(
        local.github_repository,
        local.issue_number,
        local.project_number,
        token=os.getenv("PROJECT_READ_TOKEN"),
    )
    artifact = audit_governance_drift(
        base=base,
        local=local,
        issue_payload=issue,
        project_check=project,
        baseline_provenance=baseline_provenance,
        observed_at=datetime.now(UTC),
    )

    repository = Path.cwd().resolve()
    local_root = (repository / ".conductor/local").resolve()
    output = (repository / args.output).resolve()
    if not output.is_relative_to(local_root):
        raise ValueError("drift artifacts may only be written under .conductor/local")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    return governance_audit_exit_code(artifact)


if __name__ == "__main__":
    raise SystemExit(main())
