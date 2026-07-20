"""Read-only GitHub governance drift acquisition and artifact tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request

import pytest

from vop_poc_nz.concerns import GitHubSyncPayload
from vop_poc_nz.github_drift_auditor import (
    PROJECT_FIELDS_QUERY,
    PROJECT_QUERY,
    BaselineProvenance,
    ProjectCheck,
    _load_response,
    assert_query_only,
    audit_governance_drift,
    fetch_issue,
    fetch_project_check,
    governance_audit_exit_code,
    governance_baseline_from_json,
    issue_snapshot_from_api,
)
from vop_poc_nz.github_sync_planner import GitHubIssueSnapshot

MARKER = "vop-voiage-governance-id:CON-SHR-0013"


def _provenance(*, trusted: bool = True) -> BaselineProvenance:
    return BaselineProvenance(
        schema_version="1.0.0",
        trust_state=(
            "verified_last_applied" if trusted else "unverified_initial_snapshot"
        ),
        capture_method="github_api" if trusted else "legacy_import",
        captured_at_utc="2026-07-19T00:00:00+00:00" if trusted else None,
        source_revision='W/"issue-etag"' if trusted else None,
        captured_by="governance-baseline-capture",
    )


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
    assert_query_only("project.items.read", PROJECT_QUERY)
    assert_query_only("project.fields.read", PROJECT_FIELDS_QUERY)
    with pytest.raises(ValueError, match="query-only"):
        assert_query_only(
            "project.items.read", "mutation { updateProjectV2ItemFieldValue }"
        )
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


def test_issue_api_newline_style_is_transport_noise() -> None:
    snapshot = issue_snapshot_from_api(
        _issue_api(_body().replace("\n", "\r\n")), base=_base()
    )
    assert snapshot.body == _base().body


def test_crlf_remote_body_does_not_create_governance_drift() -> None:
    artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(_body().replace("\n", "\r\n")),
        project_check=ProjectCheck.checked(_base().project_fields),
        baseline_provenance=_provenance(),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert artifact["plan"]["outcome"] == "clean"
    assert artifact["reconciliation_required"] is False


def test_missing_project_credential_is_explicit_and_never_claims_full_clean() -> None:
    artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(),
        project_check=ProjectCheck.not_checked("credential_gate"),
        baseline_provenance=_provenance(),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert artifact["network_mutation"] is False
    assert artifact["project_check"] == {
        "status": "not_checked",
        "reason": "credential_gate",
        "project_fields": [],
        "projection": "managed_fields_only",
    }
    assert artifact["scope"] == "issue_only"
    assert artifact["reconciliation_required"] is True
    assert artifact["plan"]["outcome"] == "clean"
    assert governance_audit_exit_code(artifact) == 3
    rendered = json.dumps(artifact, sort_keys=True)
    assert "token" not in rendered.casefold()
    assert ".conductor/local" not in rendered


def test_checked_project_fields_participate_in_the_pure_three_way_plan() -> None:
    artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(),
        project_check=ProjectCheck.checked(_base().project_fields),
        baseline_provenance=_provenance(),
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
        baseline_provenance=_provenance(),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert changed_artifact["plan"]["outcome"] == "remote_only"
    assert changed_artifact["reconciliation_required"] is True
    assert governance_audit_exit_code(changed_artifact) == 2


def test_retained_plan_contains_only_managed_differences() -> None:
    private_human_text = "PRIVATE-HUMAN-NOTE-DO-NOT-RETAIN"
    remote_body = _body().replace("Human preface.", private_human_text)
    artifact = audit_governance_drift(
        base=_base(),
        local=_local("Updated managed summary."),
        issue_payload=_issue_api(remote_body),
        project_check=ProjectCheck.checked(
            (*_base().project_fields, ("Human Notes", private_human_text))
        ),
        baseline_provenance=_provenance(),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    rendered = json.dumps(artifact, sort_keys=True)
    assert artifact["plan"]["outcome"] == "local_only"
    assert "proposed_issue" not in artifact["plan"]
    assert private_human_text not in rendered
    assert "Updated managed summary." in rendered
    assert artifact["plan"]["managed_differences"]


def test_unverified_baseline_is_explicit_and_non_clean() -> None:
    artifact = audit_governance_drift(
        base=_base(),
        local=_local(),
        issue_payload=_issue_api(),
        project_check=ProjectCheck.checked(_base().project_fields),
        baseline_provenance=_provenance(trusted=False),
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert artifact["baseline_capture"]["trusted_for_three_way"] is False
    assert artifact["reconciliation_required"] is True
    assert governance_audit_exit_code(artifact) == 4


def test_committed_baseline_records_honest_capture_provenance() -> None:
    baseline, provenance = governance_baseline_from_json(
        Path(".github/governance-baselines/CON-SHR-0013.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline.issue_number == 41
    assert provenance.trust_state == "unverified_initial_snapshot"
    assert provenance.trusted_for_three_way is False


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
        submitted = json.loads(request.data)
        if "GovernanceProjectItems" in submitted["query"]:
            return {
                "data": {
                    "repository": {
                        "issue": {
                            "projectItems": {
                                "nodes": [
                                    {"id": "PVTI_target", "project": {"number": 28}}
                                ],
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": "items-end",
                                },
                            }
                        }
                    }
                }
            }
        return {
            "data": {
                "node": {
                    "fieldValues": {
                        "nodes": [
                            {
                                "field": {"name": "Track ID"},
                                "textValue": "C13",
                            }
                        ],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": "fields-end",
                        },
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
    assert len(requests) == 3
    assert requests[-1].method == "POST"
    submitted = json.loads(requests[-1].data)
    assert submitted["query"].lstrip().startswith("query ")
    assert "mutation" not in submitted["query"].casefold()


def test_network_loader_rejects_non_github_origins_before_opening() -> None:
    with pytest.raises(ValueError, match=r"https://api\.github\.com"):
        _load_response(Request("file:///tmp/governance.json"))
    with pytest.raises(ValueError, match=r"https://api\.github\.com"):
        _load_response(Request("https://example.com/governance.json"))


def test_project_fetcher_paginates_items_and_fields_without_truncation() -> None:
    calls: list[tuple[str, str | None]] = []

    def loader(request):
        submitted = json.loads(request.data)
        variables = submitted["variables"]
        query = submitted["query"]
        kind = "items" if "GovernanceProjectItems" in query else "fields"
        calls.append((kind, variables["cursor"]))
        if kind == "items":
            first = variables["cursor"] is None
            return {
                "data": {
                    "repository": {
                        "issue": {
                            "projectItems": {
                                "nodes": (
                                    [{"id": "other", "project": {"number": 27}}]
                                    if first
                                    else [{"id": "target", "project": {"number": 28}}]
                                ),
                                "pageInfo": {
                                    "hasNextPage": first,
                                    "endCursor": "items-2" if first else None,
                                },
                            }
                        }
                    }
                }
            }
        first = variables["cursor"] is None
        return {
            "data": {
                "node": {
                    "fieldValues": {
                        "nodes": [
                            {
                                "field": {"name": "Record ID" if first else "Track ID"},
                                "textValue": "CON-SHR-0013" if first else "C13",
                            }
                        ],
                        "pageInfo": {
                            "hasNextPage": first,
                            "endCursor": "fields-2" if first else None,
                        },
                    }
                }
            }
        }

    result = fetch_project_check(
        "edithatogo/vop_poc_nz", 41, 28, token="read", loader=loader
    )
    assert result == ProjectCheck.checked(
        (("Record ID", "CON-SHR-0013"), ("Track ID", "C13"))
    )
    assert calls == [
        ("items", None),
        ("items", "items-2"),
        ("fields", None),
        ("fields", "fields-2"),
    ]


def test_project_fetcher_rejects_missing_pagination_metadata() -> None:
    def loader(_request):
        return {"data": {"repository": {"issue": {"projectItems": {"nodes": []}}}}}

    with pytest.raises(ValueError, match="pagination metadata is incomplete"):
        fetch_project_check(
            "edithatogo/vop_poc_nz", 41, 28, token="read", loader=loader
        )


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
    assert "if: always()" in workflow
    assert "privacy-bounded" in workflow
    assert "contents: write" not in workflow
    assert "issues: write" not in workflow
    assert "pull-requests: write" not in workflow
