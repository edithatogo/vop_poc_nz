"""Read-only GitHub governance drift acquisition and artifact tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vop_poc_nz.concerns import GitHubSyncPayload
from vop_poc_nz.github_drift_auditor import (
    PROJECT_QUERY,
    ProjectCheck,
    assert_query_only,
    audit_governance_drift,
    fetch_issue,
    fetch_project_check,
    issue_snapshot_from_api,
)
from vop_poc_nz.github_sync_planner import GitHubIssueSnapshot

MARKER = "vop-voiage-governance-id:CON-SHR-0013"


def _body(summary: str = "Canonical summary.") -> str:
    return (
        f"<!-- {MARKER} -->\n"
        "Human preface.\n\n"
        "<!-- governance:begin -->\n"
        "## Concern\n\n"
        f"{summary}\n"
        "<!-- governance:end -->\n"
    )


def _local(summary: str = "Canonical summary.") -> GitHubSyncPayload:
    return GitHubSyncPayload(
        github_repository="edithatogo/vop_poc_nz",
        issue_number=41,
        desired_state="open",
        stable_marker=MARKER,
        title="[CON-SHR-0013] Domain abstractions",
        body=_body(summary),
        labels=("conductor", "moscow-must", "roadmap"),
        project_number=28,
        project_fields=(("Record ID", "CON-SHR-0013"), ("Track ID", "C13")),
    )


def _base() -> GitHubIssueSnapshot:
    local = _local()
    return GitHubIssueSnapshot(
        github_repository=local.github_repository,
        issue_number=local.issue_number,
        state=local.desired_state,
        title=local.title,
        body=local.body,
        labels=local.labels,
        project_number=local.project_number,
        project_fields=local.project_fields,
        managed_labels=local.labels,
        managed_project_field_names=("Record ID", "Track ID"),
    )


def _issue_api(body: str | None = None) -> dict[str, object]:
    local = _local()
    return {
        "number": 41,
        "state": "open",
        "title": local.title,
        "body": body or local.body,
        "labels": [{"name": label} for label in local.labels],
    }


def test_query_allowlist_rejects_mutation_and_unknown_operations() -> None:
    assert_query_only("project.read", PROJECT_QUERY)
    with pytest.raises(ValueError, match="query-only"):
        assert_query_only("project.read", "mutation { updateProjectV2ItemFieldValue }")
    with pytest.raises(ValueError, match="allowlisted"):
        assert_query_only("repository.write", "query { viewer { login } }")


def test_issue_api_is_transformed_to_the_strict_snapshot_contract() -> None:
    snapshot = issue_snapshot_from_api(_issue_api(), base=_base())
    assert snapshot.issue_number == 41
    assert snapshot.project_fields == _base().project_fields
    assert snapshot.managed_labels == _base().managed_labels

    invalid = _issue_api()
    invalid["labels"] = ["roadmap"]
    with pytest.raises(ValueError, match="label objects"):
        issue_snapshot_from_api(invalid, base=_base())


def test_missing_project_credential_is_explicit_and_never_claims_full_clean() -> None:
    artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(),
        project_check=ProjectCheck.not_checked("credential_gate"),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert artifact["network_mutation"] is False
    assert artifact["project_check"] == {
        "status": "not_checked",
        "reason": "credential_gate",
        "project_fields": [],
    }
    assert artifact["scope"] == "issue_only"
    assert artifact["reconciliation_required"] is True
    assert artifact["plan"]["outcome"] == "clean"
    rendered = json.dumps(artifact, sort_keys=True)
    assert "token" not in rendered.casefold()
    assert ".conductor/local" not in rendered


def test_checked_project_fields_participate_in_the_pure_three_way_plan() -> None:
    artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(),
        project_check=ProjectCheck.checked(_base().project_fields),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert artifact["scope"] == "issue_and_project"
    assert artifact["reconciliation_required"] is False
    assert artifact["plan"]["outcome"] == "clean"

    changed = replace(_base(), project_fields=(("Record ID", "other"),))
    changed_artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(),
        project_check=ProjectCheck.checked(changed.project_fields),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert changed_artifact["plan"]["outcome"] == "remote_only"
    assert changed_artifact["reconciliation_required"] is True


def test_mocked_network_boundary_allows_only_issue_get_and_project_query() -> None:
    requests = []

    def issue_loader(request):
        requests.append(request)
        return _issue_api()

    assert (
        fetch_issue(
            "edithatogo/vop_poc_nz", 41, token="issue-read", loader=issue_loader
        )["number"]
        == 41
    )
    assert requests[-1].method == "GET"
    assert requests[-1].full_url.endswith("/repos/edithatogo/vop_poc_nz/issues/41")

    def project_loader(request):
        requests.append(request)
        return {
            "data": {
                "repository": {
                    "issue": {
                        "projectItems": {
                            "nodes": [
                                {
                                    "project": {"number": 28},
                                    "fieldValues": {
                                        "nodes": [
                                            {
                                                "field": {"name": "Track ID"},
                                                "textValue": "C13",
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    }
                }
            }
        }

    project = fetch_project_check(
        "edithatogo/vop_poc_nz",
        41,
        28,
        token="project-read",
        loader=project_loader,
    )
    assert project == ProjectCheck.checked((("Track ID", "C13"),))
    assert requests[-1].method == "POST"
    submitted = json.loads(requests[-1].data)
    assert submitted["query"].lstrip().startswith("query ")
    assert "mutation" not in submitted["query"].casefold()


def test_project_fetcher_does_not_touch_network_without_explicit_credential() -> None:
    def forbidden_loader(_request):
        raise AssertionError("network must not be called")

    assert fetch_project_check(
        "edithatogo/vop_poc_nz",
        41,
        28,
        token=None,
        loader=forbidden_loader,
    ) == ProjectCheck.not_checked("credential_gate")


def test_workflow_is_read_only_scheduled_manual_and_retains_artifact() -> None:
    workflow = Path(".github/workflows/governance-drift.yml").read_text(
        encoding="utf-8"
    )
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow
    assert "issues: read" in workflow
    assert "PROJECT_READ_TOKEN" in workflow
    assert "scripts/audit_github_governance_drift.py" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "contents: write" not in workflow
    assert "issues: write" not in workflow
    assert "pull-requests: write" not in workflow
