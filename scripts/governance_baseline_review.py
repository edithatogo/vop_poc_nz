#!/usr/bin/env python3
"""Capture or promote content-addressed governance baseline review artifacts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from vop_poc_nz.github_drift_auditor import (
    fetch_issue,
    fetch_project_check,
    governance_baseline_from_json,
    issue_snapshot_from_api,
)
from vop_poc_nz.governance_baseline_capture import (
    build_baseline_candidate,
    candidate_digest,
    promote_baseline_candidate,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _capture(args: argparse.Namespace) -> int:
    base, _provenance = governance_baseline_from_json(
        args.base.read_text(encoding="utf-8")
    )
    if base.issue_number is None:
        raise ValueError("governance baseline must identify an issue")
    issue = fetch_issue(
        base.github_repository,
        base.issue_number,
        token=os.getenv("GITHUB_TOKEN"),
    )
    project = fetch_project_check(
        base.github_repository,
        base.issue_number,
        base.project_number,
        token=os.getenv("PROJECT_READ_TOKEN"),
    )
    if project.status != "checked":
        raise RuntimeError(f"complete Project evidence is required: {project.reason}")
    snapshot = replace(
        issue_snapshot_from_api(issue, base=base),
        project_fields=project.project_fields,
    )
    candidate = build_baseline_candidate(
        snapshot,
        source_revision=args.source_revision,
        captured_by=args.captured_by,
        workflow_identity=args.workflow_identity,
        tool_revision=args.tool_revision,
        repository=args.repository,
        workflow_path=args.workflow_path,
        workflow_ref=args.workflow_ref,
        head_branch=args.head_branch,
        run_id=args.run_id,
        observed_at=datetime.now(UTC),
    )
    _write_json(args.output, candidate)
    print(candidate_digest(candidate))
    return 0


def _promote(args: argparse.Namespace) -> int:
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a JSON object")
    capture_run = json.loads(args.capture_run_metadata.read_text(encoding="utf-8"))
    approval_history = json.loads(args.approval_history.read_text(encoding="utf-8"))
    if not isinstance(capture_run, dict):
        raise ValueError("capture run metadata must be a JSON object")
    baseline, receipt = promote_baseline_candidate(
        candidate,
        expected_candidate_sha256=args.candidate_sha256,
        capture_run_metadata=capture_run,
        approval_history=approval_history,
        approval_environment=args.approval_environment,
        approval_run=args.approval_run,
        approved_at=datetime.now(UTC),
        administrator_bypass=args.administrator_bypass,
    )
    _write_json(args.output_baseline, baseline)
    _write_json(args.output_receipt, receipt)
    print(receipt["baseline_sha256"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--base", type=Path, required=True)
    capture.add_argument("--source-revision", required=True)
    capture.add_argument("--captured-by", required=True)
    capture.add_argument("--workflow-identity", required=True)
    capture.add_argument("--tool-revision", required=True)
    capture.add_argument("--repository", required=True)
    capture.add_argument("--workflow-path", required=True)
    capture.add_argument("--workflow-ref", required=True)
    capture.add_argument("--head-branch", required=True)
    capture.add_argument("--run-id", type=int, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.set_defaults(handler=_capture)

    promote = commands.add_parser("promote")
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--candidate-sha256", required=True)
    promote.add_argument("--capture-run-metadata", type=Path, required=True)
    promote.add_argument("--approval-history", type=Path, required=True)
    promote.add_argument("--approval-environment", required=True)
    promote.add_argument("--approval-run", required=True)
    promote.add_argument("--output-baseline", type=Path, required=True)
    promote.add_argument("--output-receipt", type=Path, required=True)
    promote.add_argument("--administrator-bypass", action="store_true")
    promote.set_defaults(handler=_promote)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
