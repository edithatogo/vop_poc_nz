"""Three-way conflict and privacy tests for GitHub governance planning."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Literal

import pytest

from vop_poc_nz.concerns import GitHubSyncPayload
from vop_poc_nz.github_sync_planner import (
    GitHubIssueSnapshot,
    issue_snapshot_from_json,
    plan_github_sync,
    sync_plan_json,
)

MARKER = "vop-voiage-governance-id:CON-SHR-0013"


def _snapshot_json_payload() -> dict[str, object]:
    return {
        "github_repository": "edithatogo/vop_poc_nz",
        "issue_number": 41,
        "state": "open",
        "title": "[CON-SHR-0013] Domain abstractions",
        "body": "bounded governance body",
        "labels": ["conductor", "roadmap"],
        "project_number": 28,
        "project_fields": [["Record ID", "CON-SHR-0013"], ["Track ID", "C13"]],
    }


def test_issue_snapshot_json_round_trip_preserves_every_contract_field() -> None:
    payload = {
        **_snapshot_json_payload(),
        "managed_labels": ["conductor"],
        "managed_project_field_names": ["Record ID"],
    }

    assert issue_snapshot_from_json(json.dumps(payload)) == GitHubIssueSnapshot(
        github_repository="edithatogo/vop_poc_nz",
        issue_number=41,
        state="open",
        title="[CON-SHR-0013] Domain abstractions",
        body="bounded governance body",
        labels=("conductor", "roadmap"),
        project_number=28,
        project_fields=(("Record ID", "CON-SHR-0013"), ("Track ID", "C13")),
        managed_labels=("conductor",),
        managed_project_field_names=("Record ID",),
    )


def test_issue_snapshot_json_defaults_optional_managed_fields() -> None:
    snapshot = issue_snapshot_from_json(json.dumps(_snapshot_json_payload()))
    assert snapshot.managed_labels == ()
    assert snapshot.managed_project_field_names == ()


@pytest.mark.parametrize("payload", [[], "snapshot", 1, None])
def test_issue_snapshot_json_requires_an_object(payload: object) -> None:
    with pytest.raises(ValueError, match=r"^GitHub issue snapshot must be an object$"):
        issue_snapshot_from_json(json.dumps(payload))


def test_issue_snapshot_json_rejects_missing_and_unexpected_fields() -> None:
    missing = _snapshot_json_payload()
    del missing["body"]
    with pytest.raises(
        ValueError,
        match=r"^GitHub issue snapshot has missing or unexpected fields$",
    ):
        issue_snapshot_from_json(json.dumps(missing))

    unexpected = {**_snapshot_json_payload(), "private_checkout": "C:/private"}
    with pytest.raises(
        ValueError,
        match=r"^GitHub issue snapshot has missing or unexpected fields$",
    ):
        issue_snapshot_from_json(json.dumps(unexpected))


@pytest.mark.parametrize("field", ["github_repository", "title", "body"])
def test_issue_snapshot_json_requires_exact_strings(field: str) -> None:
    payload = _snapshot_json_payload()
    payload[field] = 1
    with pytest.raises(
        ValueError,
        match=r"^snapshot repository, title, and body must be strings$",
    ):
        issue_snapshot_from_json(json.dumps(payload))


@pytest.mark.parametrize("state", ["pending", "OPEN", 1, True, None])
def test_issue_snapshot_json_rejects_invalid_state(state: object) -> None:
    payload = _snapshot_json_payload()
    payload["state"] = state
    with pytest.raises(ValueError, match=r"^snapshot state must be open or closed$"):
        issue_snapshot_from_json(json.dumps(payload))


@pytest.mark.parametrize("field", ["issue_number", "project_number"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", [], {}])
def test_issue_snapshot_json_rejects_non_positive_exact_integers(
    field: str, value: object
) -> None:
    payload = _snapshot_json_payload()
    payload[field] = value
    with pytest.raises(
        ValueError,
        match=rf"^snapshot {field} must be a positive integer or null$",
    ):
        issue_snapshot_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field", ["labels", "managed_labels", "managed_project_field_names"]
)
@pytest.mark.parametrize("value", ["label", [1], [True], ["valid", 1], {}])
def test_issue_snapshot_json_requires_string_arrays(field: str, value: object) -> None:
    payload = _snapshot_json_payload()
    payload[field] = value
    with pytest.raises(
        ValueError,
        match=rf"^snapshot {field} must be an array of strings$",
    ):
        issue_snapshot_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "value",
    ["fields", {}, ["Name"], [["Name"]], [["Name", "Value", "Extra"]], [[1, 2]]],
)
def test_issue_snapshot_json_requires_project_field_string_pairs(
    value: object,
) -> None:
    payload = _snapshot_json_payload()
    payload["project_fields"] = value
    with pytest.raises(
        ValueError,
        match=r"^snapshot project_fields must contain string pairs$",
    ):
        issue_snapshot_from_json(json.dumps(payload))


def _body(summary: str = "Canonical summary.") -> str:
    return (
        f"<!-- {MARKER} -->\n"
        "Human preface that the planner must preserve.\n\n"
        "<!-- governance:begin -->\n"
        "## Concern\n\n"
        f"{summary}\n"
        "<!-- governance:end -->\n\n"
        "Human follow-up that the planner must preserve.\n"
    )


def _payload(
    summary: str = "Canonical summary.",
    *,
    state: Literal["open", "closed"] = "open",
) -> GitHubSyncPayload:
    return GitHubSyncPayload(
        github_repository="edithatogo/vop_poc_nz",
        issue_number=41,
        desired_state=state,
        stable_marker=MARKER,
        title="[CON-SHR-0013] Domain abstractions",
        body=_body(summary),
        labels=("conductor", "moscow-must", "roadmap"),
        project_number=28,
        project_fields=(
            ("Record ID", "CON-SHR-0013"),
            ("Record Type", "Concern"),
            ("Track ID", "C13"),
        ),
    )


def _snapshot(
    payload: GitHubSyncPayload, *, body: str | None = None
) -> GitHubIssueSnapshot:
    return GitHubIssueSnapshot(
        github_repository=payload.github_repository,
        issue_number=payload.issue_number,
        state=payload.desired_state,
        title=payload.title,
        body=payload.body if body is None else body,
        labels=payload.labels,
        project_number=payload.project_number,
        project_fields=payload.project_fields,
    )


def test_clean_plan_ignores_human_only_body_edits() -> None:
    local = _payload()
    base = _snapshot(local)
    remote = replace(
        base,
        body=base.body.replace("Human preface", "Human edited preface"),
        labels=(*base.labels, "human-note"),
    )

    plan = plan_github_sync(base=base, local=local, remote=remote)

    assert plan.outcome == "clean"
    assert plan.proposed_issue is None
    assert plan.base_digest == plan.local_digest == plan.remote_digest


def test_local_only_replaces_bounded_section_and_preserves_human_content() -> None:
    base_payload = _payload()
    base = _snapshot(base_payload)
    remote = replace(base, labels=(*base.labels, "human-note"))
    local = _payload("Updated canonical summary.")

    plan = plan_github_sync(base=base, local=local, remote=remote)

    assert plan.outcome == "local_only"
    assert plan.proposed_issue is not None
    assert "Updated canonical summary." in plan.proposed_issue.body
    assert "Human preface that the planner must preserve." in plan.proposed_issue.body
    assert "Human follow-up that the planner must preserve." in plan.proposed_issue.body
    assert plan.proposed_issue.labels == (
        "conductor",
        "human-note",
        "moscow-must",
        "roadmap",
    )


def test_remote_only_and_conflict_fail_closed_without_a_payload() -> None:
    base_payload = _payload()
    base = _snapshot(base_payload)
    remote_only = replace(base, title="Human changed managed title")

    remote_plan = plan_github_sync(base=base, local=base_payload, remote=remote_only)
    assert remote_plan.outcome == "remote_only"
    assert remote_plan.proposed_issue is None

    local_changed = _payload("Local change.")
    conflict = plan_github_sync(base=base, local=local_changed, remote=remote_only)
    assert conflict.outcome == "conflict"
    assert conflict.proposed_issue is None


def test_marker_mismatch_and_unapproved_close_are_refused() -> None:
    local = _payload()
    base = _snapshot(local)
    wrong_marker = replace(base, body=base.body.replace("CON-SHR-0013", "CON-SHR-9999"))
    marker_plan = plan_github_sync(base=base, local=local, remote=wrong_marker)
    assert marker_plan.outcome == "conflict"
    assert marker_plan.reason == "stable_marker_mismatch"

    close = _payload(state="closed")
    close_plan = plan_github_sync(base=base, local=close, remote=base)
    assert close_plan.outcome == "conflict"
    assert close_plan.reason == "close_requires_approval"
    approved = plan_github_sync(
        base=base, local=close, remote=base, close_approved=True
    )
    assert approved.outcome == "local_only"
    assert approved.proposed_issue is not None
    assert approved.proposed_issue.state == "closed"


def test_plan_json_is_deterministic_and_contains_no_private_path() -> None:
    base_payload = _payload()
    plan = plan_github_sync(
        base=_snapshot(base_payload),
        local=_payload("Updated canonical summary."),
        remote=_snapshot(base_payload),
    )

    first = sync_plan_json(plan)
    second = sync_plan_json(plan)
    assert first == second
    assert first.endswith("\n")
    assert ".conductor/local" not in first
    assert '"network_mutation": false' in first
