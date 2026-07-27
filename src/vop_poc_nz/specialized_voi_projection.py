"""Validate and dispatch the bounded C16 specialized-VOI projection.

Network dispatch is opt-in.  Validation and dispatch-plan construction are
pure so callers can test governance inputs without credentials or mutation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

_EVENT_TYPE = "canonical-specialized-voi-updated"
_REQUIRED_POLICY = {
    "stable_markers_required": True,
    "bounded_managed_sections_only": True,
    "preserve_human_content": True,
    "three_way_conflict_detection": True,
    "fail_closed_on_missing_credentials": True,
    "new_repositories_require_explicit_registration": True,
    "automatic_merge": False,
    "automatic_issue_closure": False,
    "automatic_release": False,
}


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_registered_repositories(value: Mapping[str, Any]) -> list[str]:
    registered = value.get("registered_repositories")
    if not isinstance(registered, list) or not registered:
        raise ValueError("registered_repositories must be a non-empty list")
    repositories: list[str] = []
    for entry in registered:
        if not isinstance(entry, Mapping):
            raise ValueError("each registered repository must be an object")
        repository = _require_string(entry.get("repository"), "repository")
        if entry.get("managed_projection") is not True:
            raise ValueError(f"{repository} is not explicitly managed")
        repositories.append(repository)
    if len(repositories) != len(set(repositories)):
        raise ValueError("registered repositories must be unique")
    return repositories


def _validate_issues(value: Mapping[str, Any], repositories: list[str]) -> None:
    issues = value.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError("issues must be a non-empty list")
    for issue in issues:
        if not isinstance(issue, Mapping):
            raise ValueError("each issue must be an object")
        if _require_string(issue.get("repository"), "issue repository") not in repositories:
            raise ValueError("every issue repository must be explicitly registered")
        if not isinstance(issue.get("number"), int) or issue["number"] <= 0:
            raise ValueError("issue number must be a positive integer")


def load_projection(path: Path) -> dict[str, Any]:
    """Load and validate the stable, minimal C16 projection contract."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("projection root must be an object")
    expected = {
        "schema_version": "1.0.0",
        "projection_id": "specialized-voi-v1.2.0",
        "contract_version": "v1.2.0",
        "canonical_repository": "edithatogo/vop_poc_nz",
        "canonical_track": "C16",
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"{name} must equal {expected_value!r}")

    policy = value.get("sync_policy")
    if not isinstance(policy, Mapping) or dict(policy) != _REQUIRED_POLICY:
        raise ValueError("sync_policy does not preserve the C16 fail-closed boundary")

    repositories = _validate_registered_repositories(value)
    _validate_issues(value, repositories)
    return value


def dispatch_plan(projection: Mapping[str, Any], canonical_ref: str) -> dict[str, Any]:
    """Return a deterministic, credential-free repository-dispatch plan."""
    canonical_ref = _require_string(canonical_ref, "canonical_ref")
    targets = [entry["repository"] for entry in projection["registered_repositories"]]
    return {
        "event_type": _EVENT_TYPE,
        "targets": targets,
        "client_payload": {
            "projection_id": projection["projection_id"],
            "contract_version": projection["contract_version"],
            "canonical_repository": projection["canonical_repository"],
            "canonical_track": projection["canonical_track"],
            "canonical_ref": canonical_ref,
            "projection_path": (
                "conductor/tracks/specialized-voi-v1-2_20260727/projection.json"
            ),
        },
    }


def dispatch(plan: Mapping[str, Any], token: str | None = None) -> None:
    """Send a repository_dispatch only with an explicit credential.

    The function never broadens the target set beyond the validated plan and
    deliberately does not create merges, releases, or issue-state changes.
    """
    token = token or os.environ.get("GOVERNANCE_SYNC_TOKEN")
    if not token:
        raise RuntimeError("GOVERNANCE_SYNC_TOKEN is required for dispatch")
    body = json.dumps(
        {"event_type": plan["event_type"], "client_payload": plan["client_payload"]}
    ).encode("utf-8")
    for repository in plan["targets"]:
        request = Request(
            f"https://api.github.com/repos/{repository}/dispatches",
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            if response.status != 204:
                raise RuntimeError(f"dispatch to {repository} returned {response.status}")
