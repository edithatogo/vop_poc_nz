"""Review-gated governance baseline capture and promotion tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from vop_poc_nz.github_sync_planner import GitHubIssueSnapshot
from vop_poc_nz.governance_baseline_capture import (
    build_baseline_candidate,
    candidate_digest,
    promote_baseline_candidate,
    validate_baseline_candidate,
    validate_capture_run,
)


def _snapshot() -> GitHubIssueSnapshot:
    return GitHubIssueSnapshot(
        github_repository="edithatogo/vop_poc_nz",
        issue_number=41,
        state="open",
        title="Governance concern",
        body=(
            "human preface\n<!-- vop-voiage-governance-id:CON-SHR-0013 -->\n"
            "<!-- governance:begin -->managed<!-- governance:end -->\n"
        ),
        labels=("conductor", "human-label"),
        project_number=28,
        project_fields=(("Record ID", "CON-SHR-0013"), ("Human", "keep")),
        managed_labels=("conductor",),
        managed_project_field_names=("Record ID",),
    )


def _candidate() -> dict[str, object]:
    return build_baseline_candidate(
        _snapshot(),
        source_revision="a" * 40,
        captured_by="capture-bot",
        workflow_identity="github:edithatogo/vop_poc_nz/actions/runs/123",
        tool_revision="c" * 40,
        repository="edithatogo/vop_poc_nz",
        workflow_path=".github/workflows/governance-baseline-capture.yml",
        run_id=123,
        observed_at=datetime(2026, 7, 20, 1, 2, 3, tzinfo=UTC),
    )


def _capture_run() -> dict[str, object]:
    return {
        "id": 123,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "head_sha": "c" * 40,
        "path": ".github/workflows/governance-baseline-capture.yml",
        "repository": {"full_name": "edithatogo/vop_poc_nz"},
        "actor": {"login": "capture-bot"},
    }


def _approval_history(reviewer: str = "independent-reviewer") -> list[object]:
    return [
        {
            "state": "approved",
            "user": {"login": reviewer},
            "environments": [{"name": "governance-baseline-approval"}],
        }
    ]


def test_capture_is_untrusted_content_addressed_and_read_only() -> None:
    candidate = _candidate()

    assert validate_baseline_candidate(candidate) == candidate
    assert candidate["review"]["status"] == "pending"
    assert candidate["review"]["approval_required"] is True
    assert candidate["network_mutation"] is False
    assert candidate["integrity"] == {
        "algorithm": "sha256",
        "candidate_sha256": candidate_digest(candidate),
    }
    assert "baseline_capture" not in candidate["snapshot"]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("capture", "source_revision"), "main", "source revision"),
        (("review", "status"), "approved", "pending review"),
        (("network_mutation",), True, "read-only"),
        (("snapshot", "issue_number"), 42, "digest"),
    ],
)
def test_candidate_validation_fails_closed_on_tampering(
    path: tuple[str, ...], value: object, match: str
) -> None:
    candidate = deepcopy(_candidate())
    target = candidate
    for part in path[:-1]:
        target = target[part]  # type: ignore[assignment,index]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        validate_baseline_candidate(candidate)


def test_promotion_requires_independent_explicit_review_and_emits_receipt() -> None:
    candidate = _candidate()
    baseline, receipt = promote_baseline_candidate(
        candidate,
        expected_candidate_sha256=candidate_digest(candidate),
        capture_run_metadata=_capture_run(),
        approval_history=_approval_history(),
        approval_environment="governance-baseline-approval",
        approval_run="github:edithatogo/vop_poc_nz/actions/runs/456",
        approved_at=datetime(2026, 7, 20, 2, 3, 4, tzinfo=UTC),
    )

    assert baseline["baseline_capture"] == {
        "schema_version": "1.0.0",
        "trust_state": "verified_last_applied",
        "capture_method": "github_api",
        "captured_at_utc": "2026-07-20T01:02:03+00:00",
        "source_revision": "a" * 40,
        "captured_by": "github-environment:independent-reviewer",
    }
    assert receipt["approval"]["reviewers"] == ["independent-reviewer"]
    assert receipt["approval"]["environment"] == "governance-baseline-approval"
    assert receipt["candidate_sha256"] == candidate_digest(candidate)
    assert receipt["network_mutation"] is False


@pytest.mark.parametrize(
    ("reviewer", "digest", "match"),
    [
        ("capture-bot", None, "independent"),
        ("independent-reviewer", "0" * 64, "digest"),
        ("", None, "reviewer"),
    ],
)
def test_promotion_rejects_missing_separation_or_digest_binding(
    reviewer: str, digest: str | None, match: str
) -> None:
    candidate = _candidate()
    with pytest.raises(ValueError, match=match):
        promote_baseline_candidate(
            candidate,
            expected_candidate_sha256=digest or candidate_digest(candidate),
            capture_run_metadata=_capture_run(),
            approval_history=_approval_history(reviewer),
            approval_environment="governance-baseline-approval",
            approval_run="github:edithatogo/vop_poc_nz/actions/runs/456",
            approved_at=datetime(2026, 7, 20, 2, 3, 4, tzinfo=UTC),
        )


@pytest.mark.parametrize("field", ["id", "head_sha", "path", "actor"])
def test_capture_run_metadata_is_authoritative(field: str) -> None:
    run = deepcopy(_capture_run())
    run[field] = "forged"
    with pytest.raises(ValueError, match="capture run"):
        validate_capture_run(_candidate(), run)


def test_promotion_requires_recorded_environment_approval() -> None:
    with pytest.raises(ValueError, match="environment reviewer"):
        promote_baseline_candidate(
            _candidate(),
            expected_candidate_sha256=candidate_digest(_candidate()),
            capture_run_metadata=_capture_run(),
            approval_history=[],
            approval_environment="governance-baseline-approval",
            approval_run="github:edithatogo/vop_poc_nz/actions/runs/456",
            approved_at=datetime(2026, 7, 20, 2, 3, 4, tzinfo=UTC),
        )
