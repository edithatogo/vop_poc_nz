"""Read-only acquisition and reconciliation for GitHub governance drift."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Literal, cast
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .concerns import GitHubSyncPayload
from .critical_invariants import exact_positive_int_or_none
from .github_sync_planner import (
    GitHubIssueSnapshot,
    issue_snapshot_from_json,
    managed_local_projection,
    managed_snapshot_projection,
    plan_github_sync,
)

READ_ONLY_OPERATIONS = frozenset(
    {"issue.read", "project.items.read", "project.fields.read"}
)
_REPOSITORY_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+)$")

PROJECT_QUERY = """
query GovernanceProjectItems(
  $owner: String!,
  $name: String!,
  $issueNumber: Int!,
  $cursor: String
) {
  repository(owner: $owner, name: $name) {
    issue(number: $issueNumber) {
      projectItems(first: 100, after: $cursor) {
        nodes {
          id
          project { ... on ProjectV2 { number } }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
""".strip()

PROJECT_FIELDS_QUERY = """
query GovernanceProjectFieldValues($itemId: ID!, $cursor: String) {
  node(id: $itemId) {
    ... on ProjectV2Item {
      fieldValues(first: 100, after: $cursor) {
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
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
""".strip()


@dataclass(frozen=True)
class BaselineProvenance:
    """Evidence describing whether a snapshot is trusted as last-applied state."""

    schema_version: Literal["1.0.0"]
    trust_state: Literal["verified_last_applied", "unverified_initial_snapshot"]
    capture_method: Literal["github_api", "legacy_import"]
    captured_at_utc: str | None
    source_revision: str | None
    captured_by: str

    @property
    def trusted_for_three_way(self) -> bool:
        return self.trust_state == "verified_last_applied"


def _validate_provenance_evidence(provenance: BaselineProvenance) -> None:
    if provenance.captured_at_utc is not None:
        observed = datetime.fromisoformat(provenance.captured_at_utc)
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("baseline_capture time must be timezone-aware")
    if provenance.trusted_for_three_way and (
        provenance.capture_method != "github_api"
        or provenance.captured_at_utc is None
        or not provenance.source_revision
    ):
        raise ValueError("verified baseline_capture requires API revision evidence")


def _baseline_provenance(capture: object) -> BaselineProvenance:
    if not isinstance(capture, Mapping):
        raise ValueError("governance baseline requires baseline_capture metadata")
    expected = {
        "schema_version",
        "trust_state",
        "capture_method",
        "captured_at_utc",
        "source_revision",
        "captured_by",
    }
    if set(capture) != expected:
        raise ValueError("baseline_capture has missing or unexpected fields")
    schema_version = capture.get("schema_version")
    trust_state = capture.get("trust_state")
    capture_method = capture.get("capture_method")
    captured_at = capture.get("captured_at_utc")
    source_revision = capture.get("source_revision")
    captured_by = capture.get("captured_by")
    if schema_version != "1.0.0":
        raise ValueError("unsupported baseline_capture schema")
    if not isinstance(trust_state, str) or trust_state not in {
        "verified_last_applied",
        "unverified_initial_snapshot",
    }:
        raise ValueError("baseline_capture trust or method is invalid")
    if not isinstance(capture_method, str) or capture_method not in {
        "github_api",
        "legacy_import",
    }:
        raise ValueError("baseline_capture trust or method is invalid")
    if (captured_at is not None and not isinstance(captured_at, str)) or (
        source_revision is not None and not isinstance(source_revision, str)
    ):
        raise ValueError("baseline_capture fields are invalid")
    if not isinstance(captured_by, str) or not captured_by:
        raise ValueError("baseline_capture captured_by is required")
    normalized_trust: Literal[
        "verified_last_applied", "unverified_initial_snapshot"
    ] = (
        "verified_last_applied"
        if trust_state == "verified_last_applied"
        else "unverified_initial_snapshot"
    )
    normalized_method: Literal["github_api", "legacy_import"] = (
        "github_api" if capture_method == "github_api" else "legacy_import"
    )
    provenance = BaselineProvenance(
        schema_version="1.0.0",
        trust_state=normalized_trust,
        capture_method=normalized_method,
        captured_at_utc=captured_at,
        source_revision=source_revision,
        captured_by=captured_by,
    )
    _validate_provenance_evidence(provenance)
    return provenance


def governance_baseline_from_json(
    content: str,
) -> tuple[GitHubIssueSnapshot, BaselineProvenance]:
    """Parse a snapshot together with explicit, fail-closed capture provenance."""
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("GitHub governance baseline must be an object")
    provenance = _baseline_provenance(raw.pop("baseline_capture", None))
    snapshot = issue_snapshot_from_json(json.dumps(raw))
    return snapshot, provenance


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
    baseline_provenance: BaselineProvenance,
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
    trusted_base = baseline_provenance.trusted_for_three_way
    reconciliation_required = (
        plan.outcome != "clean" or not full_scope or not trusted_base
    )
    managed_labels = frozenset((*base.managed_labels, *local.labels))
    managed_fields = frozenset(
        (*base.managed_project_field_names, *(name for name, _ in local.project_fields))
    )
    project_artifact = project_check.as_dict()
    project_artifact["project_fields"] = [
        list(field)
        for field in project_check.project_fields
        if field[0] in managed_fields
    ]
    project_artifact["projection"] = "managed_fields_only"
    base_projection = managed_snapshot_projection(base, managed_labels, managed_fields)
    local_projection = managed_local_projection(local)
    remote_projection = managed_snapshot_projection(
        remote, managed_labels, managed_fields
    )
    differences: list[dict[str, object]] = []
    if (
        base_projection is not None
        and local_projection is not None
        and remote_projection is not None
    ):
        for field in sorted(
            base_projection.keys() | local_projection.keys() | remote_projection.keys()
        ):
            values = (
                base_projection.get(field),
                local_projection.get(field),
                remote_projection.get(field),
            )
            if len({json.dumps(value, sort_keys=True) for value in values}) > 1:
                differences.append(
                    {
                        "field": field,
                        "base": values[0],
                        "local": values[1],
                        "remote": values[2],
                    }
                )
    safe_plan = {
        key: value for key, value in asdict(plan).items() if key != "proposed_issue"
    }
    safe_plan["managed_differences"] = differences
    safe_plan["proposed_managed_projection"] = (
        local_projection if plan.outcome == "local_only" else None
    )
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
        "project_check": project_artifact,
        "baseline_capture": {
            **asdict(baseline_provenance),
            "trusted_for_three_way": trusted_base,
        },
        "plan": safe_plan,
        "reconciliation_required": reconciliation_required,
        "approval_required": True,
        "network_mutation": False,
    }


def governance_audit_exit_code(artifact: Mapping[str, object]) -> int:
    """Return distinct success, drift, incomplete-scope, and untrusted-base exits."""
    project = artifact.get("project_check")
    if not isinstance(project, Mapping) or project.get("status") != "checked":
        return 3
    baseline = artifact.get("baseline_capture")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("trusted_for_three_way") is not True
    ):
        return 4
    plan = artifact.get("plan")
    if not isinstance(plan, Mapping):
        raise RuntimeError("drift artifact plan must be an object")
    return 0 if plan.get("outcome") == "clean" else 2


def _load_response(request: Request, *, timeout: int = 20) -> Mapping[str, object]:
    endpoint = urlsplit(request.full_url)
    if endpoint.scheme != "https" or endpoint.hostname != "api.github.com":
        raise ValueError("GitHub requests require the https://api.github.com origin")
    # The exact HTTPS GitHub origin is validated above before opening.
    with urlopen(request, timeout=timeout) as response:  # nosec B310
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


def _connection_page(
    connection: Mapping[str, object], *, name: str
) -> tuple[list[object], str | None]:
    nodes = connection.get("nodes")
    page_info = connection.get("pageInfo")
    if not isinstance(nodes, list) or not isinstance(page_info, Mapping):
        raise ValueError(f"GitHub {name} pagination metadata is incomplete")
    has_next = page_info.get("hasNextPage")
    end_cursor = page_info.get("endCursor")
    if type(has_next) is not bool or (
        end_cursor is not None and not isinstance(end_cursor, str)
    ):
        raise ValueError(f"GitHub {name} pagination metadata is invalid")
    if has_next and not end_cursor:
        raise ValueError(f"GitHub {name} pagination cursor is missing")
    return cast("list[object]", nodes), end_cursor if has_next else None


def _field_values(nodes: list[object]) -> list[tuple[str, str]]:
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


def _project_items_page(
    payload: Mapping[str, object], project_number: int
) -> tuple[list[str], str | None]:
    data = _required_mapping(payload, "data")
    repository = _required_mapping(data, "repository")
    issue = _required_mapping(repository, "issue")
    project_items = _required_mapping(issue, "projectItems")
    nodes, cursor = _connection_page(project_items, name="Project items")
    item_ids: list[str] = []
    for item in nodes:
        if not isinstance(item, Mapping) or item.get("project") != {
            "number": project_number
        }:
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("GitHub Project item identity is missing")
        item_ids.append(item_id)
    return item_ids, cursor


def _project_fields_page(
    payload: Mapping[str, object],
) -> tuple[list[tuple[str, str]], str | None]:
    data = _required_mapping(payload, "data")
    node = _required_mapping(data, "node")
    field_values = _required_mapping(node, "fieldValues")
    nodes, cursor = _connection_page(field_values, name="Project field values")
    return _field_values(nodes), cursor


def _graphql_request(
    query: str,
    variables: Mapping[str, object],
    *,
    token: str,
) -> Request:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    return Request(
        "https://api.github.com/graphql",
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _project_payload(
    request: Request,
    loader: Callable[[Request], Mapping[str, object]] | None,
) -> Mapping[str, object]:
    payload = (loader or _load_response)(request)
    if payload.get("errors"):
        raise ValueError("GitHub Project query returned errors")
    return payload


def _fetch_project_item_ids(
    *,
    owner: str,
    name: str,
    issue: int,
    project_number: int,
    token: str,
    loader: Callable[[Request], Mapping[str, object]] | None,
) -> list[str]:
    item_ids: list[str] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        payload = _project_payload(
            _graphql_request(
                PROJECT_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "issueNumber": issue,
                    "cursor": cursor,
                },
                token=token,
            ),
            loader,
        )
        page_items, next_cursor = _project_items_page(payload, project_number)
        item_ids.extend(page_items)
        if next_cursor is None:
            return item_ids
        if next_cursor in seen_cursors:
            raise ValueError("GitHub Project item pagination cursor repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _fetch_project_fields(
    *,
    item_id: str,
    token: str,
    loader: Callable[[Request], Mapping[str, object]] | None,
) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        payload = _project_payload(
            _graphql_request(
                PROJECT_FIELDS_QUERY,
                {"itemId": item_id, "cursor": cursor},
                token=token,
            ),
            loader,
        )
        page_fields, next_cursor = _project_fields_page(payload)
        fields.extend(page_fields)
        if next_cursor is None:
            return fields
        if next_cursor in seen_cursors:
            raise ValueError("GitHub Project field pagination cursor repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


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
    assert_query_only("project.items.read", PROJECT_QUERY)
    assert_query_only("project.fields.read", PROJECT_FIELDS_QUERY)
    item_ids = _fetch_project_item_ids(
        owner=owner,
        name=name,
        issue=issue,
        project_number=number,
        token=token,
        loader=loader,
    )
    if len(item_ids) > 1:
        raise ValueError("issue has duplicate items in the managed GitHub Project")
    if not item_ids:
        return ProjectCheck.checked(())

    fields = _fetch_project_fields(item_id=item_ids[0], token=token, loader=loader)
    return ProjectCheck.checked(tuple(sorted(fields)))


__all__ = [
    "PROJECT_FIELDS_QUERY",
    "PROJECT_QUERY",
    "BaselineProvenance",
    "ProjectCheck",
    "assert_query_only",
    "audit_governance_drift",
    "fetch_issue",
    "fetch_project_check",
    "governance_audit_exit_code",
    "governance_baseline_from_json",
    "issue_snapshot_from_api",
]
