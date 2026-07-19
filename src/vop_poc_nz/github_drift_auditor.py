"""Read-only acquisition and reconciliation for GitHub governance drift."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Literal, cast
from urllib.request import Request, urlopen

from .concerns import GitHubSyncPayload
from .critical_invariants import exact_positive_int_or_none
from .github_sync_planner import (
    GitHubIssueSnapshot,
    issue_snapshot_from_json,
    plan_github_sync,
)

READ_ONLY_OPERATIONS = frozenset({"issue.read", "project.read"})
_REPOSITORY_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+)$")

PROJECT_QUERY = """
query GovernanceProjectFields(
  $owner: String!,
  $name: String!,
  $issueNumber: Int!
) {
  repository(owner: $owner, name: $name) {
    issue(number: $issueNumber) {
      projectItems(first: 100) {
        nodes {
          project { ... on ProjectV2 { number } }
          fieldValues(first: 100) {
            nodes {
              ... on ProjectV2ItemFieldTextValue {
                field { ... on ProjectV2Field { name } }
                textValue: text
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                field { ... on ProjectV2SingleSelectField { name } }
                selectValue: name
              }
              ... on ProjectV2ItemFieldDateValue {
                field { ... on ProjectV2Field { name } }
                dateValue: date
              }
              ... on ProjectV2ItemFieldNumberValue {
                field { ... on ProjectV2Field { name } }
                numberValue: number
              }
              ... on ProjectV2ItemFieldIterationValue {
                field { ... on ProjectV2IterationField { name } }
                iterationValue: title
              }
            }
          }
        }
      }
    }
  }
}
""".strip()


def assert_query_only(operation: str, document: str) -> None:
    """Fail closed unless an allowlisted operation contains one GraphQL query."""
    if operation not in READ_ONLY_OPERATIONS:
        raise ValueError(f"GitHub operation is not allowlisted: {operation}")
    normalized = re.sub(r"(?m)^\s*#[^\n]*", "", document).lstrip()
    if not normalized.startswith("query ") or re.search(
        r"\bmutation\b", normalized, flags=re.IGNORECASE
    ):
        raise ValueError("GitHub GraphQL operations must be query-only")


@dataclass(frozen=True)
class ProjectCheck:
    """Explicit result of the separately credentialed Project query."""

    status: Literal["checked", "not_checked"]
    reason: str
    project_fields: tuple[tuple[str, str], ...]

    @classmethod
    def checked(cls, fields: tuple[tuple[str, str], ...]) -> ProjectCheck:
        return cls(status="checked", reason="query_succeeded", project_fields=fields)

    @classmethod
    def not_checked(cls, reason: str) -> ProjectCheck:
        return cls(status="not_checked", reason=reason, project_fields=())

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "project_fields": [list(item) for item in self.project_fields],
        }


def _repository_parts(repository: str) -> tuple[str, str]:
    matched = _REPOSITORY_RE.fullmatch(repository)
    if matched is None:
        raise ValueError("GitHub repository must be an owner/name identifier")
    return matched["owner"], matched["name"]


def issue_snapshot_from_api(
    payload: Mapping[str, object], *, base: GitHubIssueSnapshot
) -> GitHubIssueSnapshot:
    """Transform a GitHub REST response through the strict snapshot parser."""
    number = exact_positive_int_or_none(payload.get("number"), field="issue number")
    if number != base.issue_number:
        raise ValueError("GitHub issue response does not match the requested issue")
    state = payload.get("state")
    title = payload.get("title")
    body = payload.get("body")
    labels = payload.get("labels")
    if type(state) is not str or state not in {"open", "closed"}:
        raise ValueError("GitHub issue state must be open or closed")
    if type(title) is not str or type(body) is not str:
        raise ValueError("GitHub issue title and body must be strings")
    if type(labels) is not list:
        raise ValueError("GitHub issue labels must be label objects")
    label_names: list[str] = []
    for item in labels:
        if type(item) is not dict:
            raise ValueError("GitHub issue labels must be label objects")
        name = item.get("name")
        if type(name) is not str:
            raise ValueError("GitHub issue labels must be label objects")
        label_names.append(name)
    strict = {
        "github_repository": base.github_repository,
        "issue_number": number,
        "state": state,
        "title": title,
        "body": body,
        "labels": label_names,
        "project_number": base.project_number,
        "project_fields": [list(item) for item in base.project_fields],
        "managed_labels": list(base.managed_labels),
        "managed_project_field_names": list(base.managed_project_field_names),
    }
    return issue_snapshot_from_json(json.dumps(strict))


def audit_governance_drift(
    *,
    base: GitHubIssueSnapshot,
    local: GitHubSyncPayload,
    issue_payload: Mapping[str, object],
    project_check: ProjectCheck,
    observed_at: datetime,
) -> dict[str, object]:
    """Build an approval-only artifact using the existing pure planner."""
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    remote = issue_snapshot_from_api(issue_payload, base=base)
    if project_check.status == "checked":
        remote = replace(remote, project_fields=project_check.project_fields)
    plan = plan_github_sync(base=base, local=local, remote=remote)
    full_scope = project_check.status == "checked"
    reconciliation_required = plan.outcome != "clean" or not full_scope
    return {
        "schema_version": "1.0.0",
        "record_id": local.stable_marker.rsplit(":", 1)[-1],
        "observed_at": observed_at.isoformat(),
        "source": {
            "github_repository": local.github_repository,
            "issue_number": local.issue_number,
            "project_number": local.project_number,
        },
        "scope": "issue_and_project" if full_scope else "issue_only",
        "issue_check": {"status": "checked", "operation": "issue.read"},
        "project_check": project_check.as_dict(),
        "plan": asdict(plan),
        "reconciliation_required": reconciliation_required,
        "approval_required": True,
        "network_mutation": False,
    }


def _load_response(request: Request, *, timeout: int = 20) -> Mapping[str, object]:
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("GitHub response must be a JSON object")
    return payload


def fetch_issue(
    repository: str,
    issue_number: int,
    *,
    token: str | None,
    loader: Callable[[Request], Mapping[str, object]] | None = None,
) -> Mapping[str, object]:
    """Perform the sole allowlisted REST GET for an issue snapshot."""
    owner, name = _repository_parts(repository)
    number = exact_positive_int_or_none(issue_number, field="issue number")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"https://api.github.com/repos/{owner}/{name}/issues/{number}",
        headers=headers,
        method="GET",
    )
    return (loader or _load_response)(request)


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError("GitHub Project response has an unexpected shape")
    return cast("Mapping[str, object]", nested)


def _project_item_fields(item: object, project_number: int) -> list[tuple[str, str]]:
    if not isinstance(item, Mapping) or item.get("project") != {
        "number": project_number
    }:
        return []
    raw_fields = item.get("fieldValues")
    if not isinstance(raw_fields, Mapping):
        raise ValueError("GitHub Project field values must be an array")
    nodes = raw_fields.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("GitHub Project field values must be an array")
    fields: list[tuple[str, str]] = []
    for raw in nodes:
        if not isinstance(raw, Mapping):
            continue
        raw_mapping = cast("Mapping[str, object]", raw)
        field = raw_mapping.get("field")
        if not isinstance(field, Mapping):
            continue
        name = field.get("name")
        value = next(
            (
                candidate
                for key in (
                    "textValue",
                    "selectValue",
                    "dateValue",
                    "numberValue",
                    "iterationValue",
                )
                if (candidate := raw_mapping.get(key)) is not None
            ),
            None,
        )
        if isinstance(name, str) and isinstance(value, (str, int, float)):
            fields.append((name, str(value)))
    return fields


def _project_fields(payload: Mapping[str, object], project_number: int) -> ProjectCheck:
    data = _required_mapping(payload, "data")
    repository = _required_mapping(data, "repository")
    issue = _required_mapping(repository, "issue")
    project_items = _required_mapping(issue, "projectItems")
    nodes = project_items.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("GitHub Project items must be an array")
    fields = [
        field for item in nodes for field in _project_item_fields(item, project_number)
    ]
    return ProjectCheck.checked(tuple(sorted(fields)))


def fetch_project_check(
    repository: str,
    issue_number: int,
    project_number: int | None,
    *,
    token: str | None,
    loader: Callable[[Request], Mapping[str, object]] | None = None,
) -> ProjectCheck:
    """Query Project fields only when an explicit read credential is supplied."""
    if not token:
        return ProjectCheck.not_checked("credential_gate")
    number = exact_positive_int_or_none(project_number, field="project number")
    if number is None:
        return ProjectCheck.not_checked("project_not_linked")
    owner, name = _repository_parts(repository)
    issue = exact_positive_int_or_none(issue_number, field="issue number")
    if issue is None:
        raise ValueError("issue number is required")
    assert_query_only("project.read", PROJECT_QUERY)
    body = json.dumps(
        {
            "query": PROJECT_QUERY,
            "variables": {
                "owner": owner,
                "name": name,
                "issueNumber": issue,
            },
        }
    ).encode("utf-8")
    request = Request(
        "https://api.github.com/graphql",
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    payload = (loader or _load_response)(request)
    if payload.get("errors"):
        raise ValueError("GitHub Project query returned errors")
    return _project_fields(payload, number)


__all__ = [
    "PROJECT_QUERY",
    "ProjectCheck",
    "assert_query_only",
    "audit_governance_drift",
    "fetch_issue",
    "fetch_project_check",
    "issue_snapshot_from_api",
]
