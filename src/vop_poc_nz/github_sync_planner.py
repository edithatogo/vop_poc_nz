"""Pure, conflict-safe three-way planning for governance GitHub projections.

This module has no network or mutation capability.  It compares the last
applied managed projection (base), the desired local projection, and a remote
snapshot, then returns a deterministic plan for a separately authorised tool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal

from .concerns import GitHubSyncPayload

SyncOutcome = Literal["clean", "local_only", "remote_only", "conflict"]

_MARKER_RE = re.compile(
    r"<!--\s*(vop-voiage-governance-id:"
    r"(?:CON|ASM|RSK|DEC|EVR|ISL)-(?:VOP|VOI|SHR)-[0-9]{4})\s*-->"
)
_SECTION_RE = re.compile(
    r"<!-- governance:begin -->.*?<!-- governance:end -->", re.DOTALL
)


@dataclass(frozen=True)
class GitHubIssueSnapshot:
    """Read-only GitHub issue and Project state supplied by a caller."""

    github_repository: str
    issue_number: int | None
    state: Literal["open", "closed"]
    title: str
    body: str
    labels: tuple[str, ...]
    project_number: int | None
    project_fields: tuple[tuple[str, str], ...]
    managed_labels: tuple[str, ...] = ()
    managed_project_field_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedIssueUpdate:
    """Desired issue state without any operation that could apply it."""

    github_repository: str
    issue_number: int | None
    state: Literal["open", "closed"]
    title: str
    body: str
    labels: tuple[str, ...]
    project_number: int | None
    project_fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GitHubSyncPlan:
    """Deterministic result of a three-way managed-projection comparison."""

    schema_version: Literal["1.0.0"]
    outcome: SyncOutcome
    reason: str
    stable_marker: str
    base_digest: str
    local_digest: str
    remote_digest: str
    preserves_human_content: Literal[True]
    network_mutation: Literal[False]
    proposed_issue: PlannedIssueUpdate | None


def _canonical_digest(value: Mapping[str, Any]) -> str:
    content = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(content).hexdigest()


def _marker(body: str) -> str | None:
    matches = _MARKER_RE.findall(body)
    return matches[0] if len(matches) == 1 else None


def _section(body: str) -> str | None:
    matches = _SECTION_RE.findall(body)
    return matches[0] if len(matches) == 1 else None


def _fallback_digest(body: str) -> str:
    return _canonical_digest({"invalid_body": body})


def _snapshot_projection(
    snapshot: GitHubIssueSnapshot,
    managed_labels: frozenset[str],
    managed_project_field_names: frozenset[str],
) -> dict[str, Any] | None:
    section = _section(snapshot.body)
    if section is None:
        return None
    return {
        "body_section": section,
        "labels": sorted(label for label in snapshot.labels if label in managed_labels),
        "project_fields": sorted(
            field
            for field in snapshot.project_fields
            if field[0] in managed_project_field_names
        ),
        "state": snapshot.state,
        "title": snapshot.title,
    }


def _local_projection(payload: GitHubSyncPayload) -> dict[str, Any] | None:
    section = _section(payload.body)
    if section is None:
        return None
    return {
        "body_section": section,
        "labels": sorted(payload.labels),
        "project_fields": sorted(payload.project_fields),
        "state": payload.desired_state,
        "title": payload.title,
    }


def _replace_section(remote_body: str, local_body: str) -> str:
    local_section = _section(local_body)
    if local_section is None or _section(remote_body) is None:
        raise ValueError("both local and remote bodies require one managed section")
    return _SECTION_RE.sub(lambda _match: local_section, remote_body, count=1)


def _proposal(
    *,
    base: GitHubIssueSnapshot,
    local: GitHubSyncPayload,
    remote: GitHubIssueSnapshot,
) -> PlannedIssueUpdate:
    managed_labels = set(base.managed_labels) | set(local.labels)
    unmanaged_labels = set(remote.labels) - managed_labels
    labels = tuple(sorted(unmanaged_labels | set(local.labels)))
    managed_field_names = set(base.managed_project_field_names) | {
        name for name, _value in local.project_fields
    }
    unmanaged_fields = {
        field for field in remote.project_fields if field[0] not in managed_field_names
    }
    project_fields = tuple(sorted(unmanaged_fields | set(local.project_fields)))
    return PlannedIssueUpdate(
        github_repository=local.github_repository,
        issue_number=local.issue_number,
        state=local.desired_state,
        title=local.title,
        body=_replace_section(remote.body, local.body),
        labels=labels,
        project_number=local.project_number,
        project_fields=project_fields,
    )


def _conflict_plan(
    *,
    reason: str,
    stable_marker: str,
    base_digest: str,
    local_digest: str,
    remote_digest: str,
) -> GitHubSyncPlan:
    return GitHubSyncPlan(
        schema_version="1.0.0",
        outcome="conflict",
        reason=reason,
        stable_marker=stable_marker,
        base_digest=base_digest,
        local_digest=local_digest,
        remote_digest=remote_digest,
        preserves_human_content=True,
        network_mutation=False,
        proposed_issue=None,
    )


def plan_github_sync(
    *,
    base: GitHubIssueSnapshot,
    local: GitHubSyncPayload,
    remote: GitHubIssueSnapshot,
    close_approved: bool = False,
) -> GitHubSyncPlan:
    """Plan a conflict-safe GitHub update without performing any mutation."""
    base_marker = _marker(base.body)
    local_marker = _marker(local.body)
    remote_marker = _marker(remote.body)
    marker = local.stable_marker
    if not (
        marker == base_marker == local_marker == remote_marker
        and base.github_repository
        == local.github_repository
        == remote.github_repository
        and base.issue_number == local.issue_number == remote.issue_number
    ):
        return _conflict_plan(
            reason="stable_marker_mismatch",
            stable_marker=marker,
            base_digest=_fallback_digest(base.body),
            local_digest=_fallback_digest(local.body),
            remote_digest=_fallback_digest(remote.body),
        )

    managed_labels = frozenset((*base.managed_labels, *local.labels))
    managed_project_field_names = frozenset(
        (*base.managed_project_field_names, *(name for name, _ in local.project_fields))
    )
    base_projection = _snapshot_projection(
        base, managed_labels, managed_project_field_names
    )
    local_projection = _local_projection(local)
    remote_projection = _snapshot_projection(
        remote, managed_labels, managed_project_field_names
    )
    if any(
        projection is None
        for projection in (base_projection, local_projection, remote_projection)
    ):
        return _conflict_plan(
            reason="managed_section_invalid",
            stable_marker=marker,
            base_digest=_fallback_digest(base.body),
            local_digest=_fallback_digest(local.body),
            remote_digest=_fallback_digest(remote.body),
        )

    assert base_projection is not None
    assert local_projection is not None
    assert remote_projection is not None
    base_digest = _canonical_digest(base_projection)
    local_digest = _canonical_digest(local_projection)
    remote_digest = _canonical_digest(remote_projection)

    if local.desired_state == "closed" and not close_approved:
        return _conflict_plan(
            reason="close_requires_approval",
            stable_marker=marker,
            base_digest=base_digest,
            local_digest=local_digest,
            remote_digest=remote_digest,
        )

    proposed: PlannedIssueUpdate | None = None
    if local_digest == remote_digest:
        outcome: SyncOutcome = "clean"
        reason = "managed_projections_match"
    elif local_digest == base_digest:
        outcome = "remote_only"
        reason = "remote_managed_projection_changed"
    elif remote_digest == base_digest:
        outcome = "local_only"
        reason = "local_managed_projection_changed"
        proposed = _proposal(base=base, local=local, remote=remote)
    else:
        outcome = "conflict"
        reason = "local_and_remote_managed_projections_changed"

    return GitHubSyncPlan(
        schema_version="1.0.0",
        outcome=outcome,
        reason=reason,
        stable_marker=marker,
        base_digest=base_digest,
        local_digest=local_digest,
        remote_digest=remote_digest,
        preserves_human_content=True,
        network_mutation=False,
        proposed_issue=proposed,
    )


def issue_snapshot_from_json(content: str) -> GitHubIssueSnapshot:
    """Parse a strict snapshot from caller-supplied JSON."""
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("GitHub issue snapshot must be an object")
    required = {
        "github_repository",
        "issue_number",
        "state",
        "title",
        "body",
        "labels",
        "project_number",
        "project_fields",
    }
    optional = {"managed_labels", "managed_project_field_names"}
    if not required <= set(raw) or set(raw) - required - optional:
        raise ValueError("GitHub issue snapshot has missing or unexpected fields")
    state = raw["state"]
    if state not in {"open", "closed"}:
        raise ValueError("snapshot state must be open or closed")
    return GitHubIssueSnapshot(
        github_repository=str(raw["github_repository"]),
        issue_number=int(raw["issue_number"])
        if raw["issue_number"] is not None
        else None,
        state=state,
        title=str(raw["title"]),
        body=str(raw["body"]),
        labels=tuple(str(item) for item in raw["labels"]),
        project_number=int(raw["project_number"])
        if raw["project_number"] is not None
        else None,
        project_fields=tuple(
            (str(item[0]), str(item[1])) for item in raw["project_fields"]
        ),
        managed_labels=tuple(str(item) for item in raw.get("managed_labels", ())),
        managed_project_field_names=tuple(
            str(item) for item in raw.get("managed_project_field_names", ())
        ),
    )


def sync_plan_json(plan: GitHubSyncPlan) -> str:
    """Serialize a plan deterministically without filesystem context."""
    return json.dumps(asdict(plan), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


__all__ = [
    "GitHubIssueSnapshot",
    "GitHubSyncPlan",
    "PlannedIssueUpdate",
    "issue_snapshot_from_json",
    "plan_github_sync",
    "sync_plan_json",
]
