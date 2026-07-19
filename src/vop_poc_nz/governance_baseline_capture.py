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
    observed_at: datetime,
) -> dict[str, object]:
    """Build an explicitly untrusted review candidate from a read-only snapshot."""
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise ValueError("source revision must be an exact lowercase Git commit SHA")
    if not captured_by.strip():
        raise ValueError("captured_by is required")
    if _WORKFLOW_RE.fullmatch(workflow_identity) is None:
        raise ValueError("workflow identity must name an exact GitHub Actions run")
    candidate: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "governance_baseline_review_candidate",
        "capture": {
            "observed_at_utc": _aware_iso(observed_at, field="observed_at"),
            "source_revision": source_revision,
            "captured_by": captured_by,
            "workflow_identity": workflow_identity,
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
        },
        field="capture",
    )
    revision = capture.get("source_revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("source revision must be an exact lowercase Git commit SHA")
    captured_by = capture.get("captured_by")
    workflow = capture.get("workflow_identity")
    observed_at = capture.get("observed_at_utc")
    if not isinstance(captured_by, str) or not captured_by.strip():
        raise ValueError("captured_by is required")
    if not isinstance(workflow, str) or _WORKFLOW_RE.fullmatch(workflow) is None:
        raise ValueError("workflow identity must name an exact GitHub Actions run")
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


def promote_baseline_candidate(
    candidate: Mapping[str, object],
    *,
    expected_candidate_sha256: str,
    approved_by: str,
    approval_run: str,
    approved_at: datetime,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create review artifacts after an external approval gate has succeeded."""
    validated = validate_baseline_candidate(candidate)
    digest = candidate_digest(validated)
    if (
        _SHA256_RE.fullmatch(expected_candidate_sha256) is None
        or expected_candidate_sha256 != digest
    ):
        raise ValueError("approved candidate digest does not match")
    capture = cast("Mapping[str, object]", validated["capture"])
    if not approved_by.strip():
        raise ValueError("reviewer identity is required")
    if approved_by == capture["captured_by"]:
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
        "captured_by": approved_by,
    }
    baseline_sha256 = sha256(_canonical_bytes(baseline)).hexdigest()
    receipt = {
        "schema_version": "1.0.0",
        "kind": "governance_baseline_approval_receipt",
        "candidate_sha256": digest,
        "baseline_sha256": baseline_sha256,
        "source_revision": capture["source_revision"],
        "approval": {
            "approved_by": approved_by,
            "approved_at_utc": approved_at_iso,
            "workflow_identity": approval_run,
            "separation_of_duties": True,
        },
        "network_mutation": False,
    }
    return baseline, receipt


__all__ = [
    "build_baseline_candidate",
    "candidate_digest",
    "promote_baseline_candidate",
    "validate_baseline_candidate",
]
