"""Content-addressed, review-gated governance baseline capture."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from typing import cast

from .github_sync_planner import GitHubIssueSnapshot, issue_snapshot_from_json

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WORKFLOW_RE = re.compile(
    r"^github:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[1-9][0-9]*$"
)
_CANDIDATE_KEYS = {
    "schema_version",
    "kind",
    "capture",
    "snapshot",
    "review",
    "network_mutation",
    "integrity",
}
_WORKFLOW_PATH = ".github/workflows/governance-baseline-capture.yml"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _aware_iso(value: datetime, *, field: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.isoformat()


def _strict_mapping(
    value: object, *, keys: set[str], field: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} has missing or unexpected fields")
    return cast("Mapping[str, object]", value)


def _snapshot_dict(snapshot: GitHubIssueSnapshot) -> dict[str, object]:
    raw = asdict(snapshot)
    raw["labels"] = list(snapshot.labels)
    raw["project_fields"] = [list(item) for item in snapshot.project_fields]
    raw["managed_labels"] = list(snapshot.managed_labels)
    raw["managed_project_field_names"] = list(snapshot.managed_project_field_names)
    return raw


def _validate_workflow_binding(
    repository: object,
    workflow_path: object,
    workflow_ref: object,
    head_branch: object,
) -> None:
    """Require one exact default-branch workflow identity across all API forms."""
    if not isinstance(repository, str) or not repository:
        raise ValueError("capture repository is required")
    if workflow_path != _WORKFLOW_PATH:
        raise ValueError("capture workflow path is not allowlisted")
    if not isinstance(head_branch, str) or not head_branch.strip():
        raise ValueError("capture head branch is required")
    expected_workflow_ref = f"{repository}/{workflow_path}@refs/heads/{head_branch}"
    if workflow_ref != expected_workflow_ref:
        raise ValueError(
            "capture workflow ref does not match repository, path, and branch"
        )


def candidate_digest(candidate: Mapping[str, object]) -> str:
    """Return the digest over every candidate field except its integrity seal."""
    body = {key: value for key, value in candidate.items() if key != "integrity"}
    return sha256(_canonical_bytes(body)).hexdigest()


def build_baseline_candidate(
    snapshot: GitHubIssueSnapshot,
    *,
    source_revision: str,
    captured_by: str,
    workflow_identity: str,
    tool_revision: str,
    repository: str,
    workflow_path: str,
    workflow_ref: str,
    head_branch: str,
    run_id: int,
    observed_at: datetime,
) -> dict[str, object]:
    """Build an explicitly untrusted review candidate from a read-only snapshot."""
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise ValueError("source revision must be an exact lowercase Git commit SHA")
    if not captured_by.strip():
        raise ValueError("captured_by is required")
    if _WORKFLOW_RE.fullmatch(workflow_identity) is None:
        raise ValueError("workflow identity must name an exact GitHub Actions run")
    if _REVISION_RE.fullmatch(tool_revision) is None:
        raise ValueError("tool revision must be an exact lowercase Git commit SHA")
    if repository != snapshot.github_repository:
        raise ValueError("capture repository must match the snapshot repository")
    _validate_workflow_binding(repository, workflow_path, workflow_ref, head_branch)
    if type(run_id) is not int or run_id < 1:
        raise ValueError("capture run ID must be a positive integer")
    candidate: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "governance_baseline_review_candidate",
        "capture": {
            "observed_at_utc": _aware_iso(observed_at, field="observed_at"),
            "source_revision": source_revision,
            "captured_by": captured_by,
            "workflow_identity": workflow_identity,
            "tool_revision": tool_revision,
            "repository": repository,
            "workflow_path": workflow_path,
            "workflow_ref": workflow_ref,
            "head_branch": head_branch,
            "run_id": run_id,
        },
        "snapshot": _snapshot_dict(snapshot),
        "review": {
            "status": "pending",
            "approval_required": True,
            "independent_reviewer_required": True,
        },
        "network_mutation": False,
    }
    candidate["integrity"] = {
        "algorithm": "sha256",
        "candidate_sha256": candidate_digest(candidate),
    }
    return candidate


def _validate_capture(value: object) -> Mapping[str, object]:
    capture = _strict_mapping(
        value,
        keys={
            "observed_at_utc",
            "source_revision",
            "captured_by",
            "workflow_identity",
            "tool_revision",
            "repository",
            "workflow_path",
            "workflow_ref",
            "head_branch",
            "run_id",
        },
        field="capture",
    )
    revision = capture.get("source_revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("source revision must be an exact lowercase Git commit SHA")
    captured_by = capture.get("captured_by")
    workflow = capture.get("workflow_identity")
    tool_revision = capture.get("tool_revision")
    repository = capture.get("repository")
    workflow_path = capture.get("workflow_path")
    workflow_ref = capture.get("workflow_ref")
    head_branch = capture.get("head_branch")
    run_id = capture.get("run_id")
    observed_at = capture.get("observed_at_utc")
    if not isinstance(captured_by, str) or not captured_by.strip():
        raise ValueError("captured_by is required")
    if not isinstance(workflow, str) or _WORKFLOW_RE.fullmatch(workflow) is None:
        raise ValueError("workflow identity must name an exact GitHub Actions run")
    if (
        not isinstance(tool_revision, str)
        or _REVISION_RE.fullmatch(tool_revision) is None
    ):
        raise ValueError("tool revision must be an exact lowercase Git commit SHA")
    _validate_workflow_binding(repository, workflow_path, workflow_ref, head_branch)
    if type(run_id) is not int or run_id < 1:
        raise ValueError("capture run ID must be a positive integer")
    if not isinstance(observed_at, str):
        raise ValueError("capture time must be a string")
    parsed_time = datetime.fromisoformat(observed_at)
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise ValueError("capture time must be timezone-aware")
    return capture


def validate_baseline_candidate(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate a candidate without changing its pending-review state."""
    candidate = _strict_mapping(value, keys=_CANDIDATE_KEYS, field="candidate")
    if (
        candidate.get("schema_version") != "1.0.0"
        or candidate.get("kind") != "governance_baseline_review_candidate"
    ):
        raise ValueError("unsupported governance baseline candidate")
    _validate_capture(candidate.get("capture"))
    review = _strict_mapping(
        candidate.get("review"),
        keys={"status", "approval_required", "independent_reviewer_required"},
        field="review",
    )
    if review != {
        "status": "pending",
        "approval_required": True,
        "independent_reviewer_required": True,
    }:
        raise ValueError("candidate must remain pending review")
    if candidate.get("network_mutation") is not False:
        raise ValueError("candidate capture must be read-only")
    snapshot = candidate.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("candidate snapshot must be an object")
    issue_snapshot_from_json(json.dumps(snapshot))
    integrity = _strict_mapping(
        candidate.get("integrity"),
        keys={"algorithm", "candidate_sha256"},
        field="integrity",
    )
    digest = integrity.get("candidate_sha256")
    if (
        integrity.get("algorithm") != "sha256"
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or digest != candidate_digest(candidate)
    ):
        raise ValueError("candidate digest does not match its content")
    return dict(candidate)


def validate_capture_run(
    candidate: Mapping[str, object], run_metadata: Mapping[str, object]
) -> None:
    """Bind a candidate to authoritative GitHub Actions run metadata."""
    validated = validate_baseline_candidate(candidate)
    capture = cast("Mapping[str, object]", validated["capture"])
    repository = run_metadata.get("repository")
    actor = run_metadata.get("actor")
    if not isinstance(repository, Mapping) or not isinstance(actor, Mapping):
        raise ValueError("capture run metadata is incomplete")
    expected = {
        "id": capture["run_id"],
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "head_sha": capture["tool_revision"],
        "path": f"{capture['workflow_path']}@{capture['head_branch']}",
        "head_branch": capture["head_branch"],
    }
    for field, value in expected.items():
        if run_metadata.get(field) != value:
            raise ValueError(f"capture run {field} does not match candidate")
    if repository.get("full_name") != capture["repository"]:
        raise ValueError("capture run repository does not match candidate")
    if actor.get("login") != capture["captured_by"]:
        raise ValueError("capture run actor does not match candidate")
    expected_identity = (
        f"github:{capture['repository']}/actions/runs/{capture['run_id']}"
    )
    if capture["workflow_identity"] != expected_identity:
        raise ValueError("capture workflow identity does not match run ID")


def _approved_reviewers(history: object, *, environment: str) -> tuple[str, ...]:
    if not isinstance(history, list):
        raise ValueError("approval history must be an array")
    reviewers: set[str] = set()
    for entry in history:
        if not isinstance(entry, Mapping) or entry.get("state") != "approved":
            continue
        environments = entry.get("environments")
        user = entry.get("user")
        if not isinstance(environments, list) or not isinstance(user, Mapping):
            continue
        login = user.get("login")
        if not isinstance(login, str) or not login:
            continue
        if any(
            isinstance(item, Mapping) and item.get("name") == environment
            for item in environments
        ):
            reviewers.add(login)
    if not reviewers:
        raise ValueError("no approved environment reviewer evidence was found")
    return tuple(sorted(reviewers, key=str.casefold))


def promote_baseline_candidate(
    candidate: Mapping[str, object],
    *,
    expected_candidate_sha256: str,
    capture_run_metadata: Mapping[str, object],
    approval_history: object,
    approval_environment: str,
    approval_run: str,
    approved_at: datetime,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create review artifacts after an external approval gate has succeeded."""
    validated = validate_baseline_candidate(candidate)
    validate_capture_run(validated, capture_run_metadata)
    digest = candidate_digest(validated)
    if (
        _SHA256_RE.fullmatch(expected_candidate_sha256) is None
        or expected_candidate_sha256 != digest
    ):
        raise ValueError("approved candidate digest does not match")
    capture = cast("Mapping[str, object]", validated["capture"])
    reviewers = _approved_reviewers(approval_history, environment=approval_environment)
    if any(
        reviewer.casefold() == str(capture["captured_by"]).casefold()
        for reviewer in reviewers
    ):
        raise ValueError("baseline approval requires an independent reviewer")
    if _WORKFLOW_RE.fullmatch(approval_run) is None:
        raise ValueError("approval run must name an exact GitHub Actions run")
    approved_at_iso = _aware_iso(approved_at, field="approved_at")
    snapshot = cast("Mapping[str, object]", validated["snapshot"])
    baseline = dict(snapshot)
    baseline["baseline_capture"] = {
        "schema_version": "1.0.0",
        "trust_state": "verified_last_applied",
        "capture_method": "github_api",
        "captured_at_utc": capture["observed_at_utc"],
        "source_revision": capture["source_revision"],
        "captured_by": "github-environment:" + ",".join(reviewers),
    }
    baseline_sha256 = sha256(_canonical_bytes(baseline)).hexdigest()
    receipt = {
        "schema_version": "1.0.0",
        "kind": "governance_baseline_approval_receipt",
        "candidate_sha256": digest,
        "baseline_sha256": baseline_sha256,
        "source_revision": capture["source_revision"],
        "approval": {
            "reviewers": list(reviewers),
            "environment": approval_environment,
            "approved_at_utc": approved_at_iso,
            "workflow_identity": approval_run,
            "separation_of_duties": True,
            "history_sha256": sha256(_canonical_bytes(approval_history)).hexdigest(),
        },
        "network_mutation": False,
    }
    return baseline, receipt


__all__ = [
    "build_baseline_candidate",
    "candidate_digest",
    "promote_baseline_candidate",
    "validate_baseline_candidate",
    "validate_capture_run",
]
