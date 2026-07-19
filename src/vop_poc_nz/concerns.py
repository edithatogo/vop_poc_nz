"""Privacy-safe governance records and deterministic GitHub projections.

The local ledger is the source of truth.  This module deliberately does not
perform network operations: it only validates records, exports JSON Schemas,
and builds payloads that a separately authorised synchronizer could consume.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RecordVisibility = Literal["public", "repository", "local_private"]
Repository = Literal["vop_poc_nz", "voiage", "shared"]
Gate = Literal[
    "none", "local", "external", "human", "credential", "hardware", "publication"
]
MosCow = Literal["must", "should", "could", "wont_now"]

_ID_RE = re.compile(r"^(CON|ASM|RSK|DEC|EVR|ISL)-(VOP|VOI|SHR)-[0-9]{4}$")
_PREFIXES = {
    "concern": "CON",
    "assumption": "ASM",
    "risk": "RSK",
    "decision": "DEC",
    "evidence_reference": "EVR",
    "issue_link": "ISL",
}


class GovernanceModel(BaseModel):
    """Strict, deeply collection-safe base for governance values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Relation(GovernanceModel):
    """Typed edge between governance records."""

    relation: Literal[
        "informs",
        "depends_on",
        "blocks",
        "mitigates",
        "mitigated_by",
        "resolved_by",
        "supersedes",
        "implements",
        "supports",
        "challenges",
    ]
    target_id: str = Field(
        pattern=r"^(CON|ASM|RSK|DEC|EVR|ISL)-(VOP|VOI|SHR)-[0-9]{4}$"
    )


class BaseGovernanceRecord(GovernanceModel):
    """Fields shared by all governance records."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    record_version: int = Field(default=1, ge=1)
    id: str
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    repository: Repository = "shared"
    track_ids: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    moscow: MosCow = "should"
    priority: Literal["P0", "P1", "P2", "P3"] = "P2"
    gate: Gate = "local"
    visibility: RecordVisibility = "public"
    owner_role: str = "maintainer"
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    relations: tuple[Relation, ...] = ()
    evidence_reference_ids: tuple[str, ...] = ()
    issue_link_ids: tuple[str, ...] = ()

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("governance timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_identity(self) -> BaseGovernanceRecord:
        match = _ID_RE.fullmatch(self.id)
        if match is None:
            raise ValueError(f"invalid governance record id: {self.id}")
        record_type = str(getattr(self, "record_type", ""))
        expected = _PREFIXES.get(record_type)
        if expected is None or match.group(1) != expected:
            raise ValueError(
                f"record id {self.id} does not match record type {record_type}"
            )
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("track_ids must be unique")
        if len(set(self.evidence_reference_ids)) != len(self.evidence_reference_ids):
            raise ValueError("evidence_reference_ids must be unique")
        if len(set(self.issue_link_ids)) != len(self.issue_link_ids):
            raise ValueError("issue_link_ids must be unique")
        return self


class Concern(BaseGovernanceRecord):
    """An unresolved question that requires investigation or disposition."""

    record_type: Literal["concern"] = "concern"
    status: Literal["open", "investigating", "monitoring", "resolved", "accepted"]
    question: str = Field(min_length=1)
    impact_if_unresolved: str = Field(min_length=1)
    resolution_criteria: tuple[str, ...] = Field(min_length=1)
    raised_by_role: str = "reviewer"


class Assumption(BaseGovernanceRecord):
    """A proposition relied upon by an analysis, contract, or operation."""

    record_type: Literal["assumption"] = "assumption"
    status: Literal["proposed", "active", "validated", "invalidated", "retired"]
    category: Literal[
        "normative",
        "structural",
        "data",
        "statistical",
        "computational",
        "operational",
        "external",
    ]
    rationale: str = Field(min_length=1)
    validation_method: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    review_due: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class Risk(BaseGovernanceRecord):
    """An uncertain event with an adverse consequence."""

    record_type: Literal["risk"] = "risk"
    status: Literal["open", "mitigating", "accepted", "realized", "closed"]
    cause: str = Field(min_length=1)
    event: str = Field(min_length=1)
    consequence: str = Field(min_length=1)
    likelihood: Literal["rare", "unlikely", "possible", "likely", "almost_certain"]
    impact: Literal["negligible", "minor", "moderate", "major", "critical"]
    risk_level: Literal["low", "medium", "high", "critical"]
    mitigations: tuple[str, ...] = ()
    residual_risk_level: Literal["low", "medium", "high", "critical"] | None = None

    @model_validator(mode="after")
    def _require_high_risk_controls(self) -> Risk:
        if self.risk_level in {"high", "critical"}:
            if not self.mitigations:
                raise ValueError("high and critical risks require a mitigation")
            if not self.issue_link_ids:
                raise ValueError("high and critical risks require an issue link")
        return self


class Decision(BaseGovernanceRecord):
    """A choice with its alternatives, rationale, and approval boundary."""

    record_type: Literal["decision"] = "decision"
    status: Literal["proposed", "accepted", "rejected", "superseded"]
    question: str = Field(min_length=1)
    options: tuple[str, ...] = Field(min_length=2)
    selected_option: str | None = None
    rationale: str = Field(min_length=1)
    consequences: tuple[str, ...] = Field(min_length=1)
    reversibility: Literal["reversible", "costly", "irreversible"]
    approved_by_role: str | None = None
    approved_at: datetime | None = None
    supersedes: tuple[str, ...] = ()

    @field_validator("approved_at")
    @classmethod
    def _approval_timestamp_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approved_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_disposition(self) -> Decision:
        if (
            self.selected_option is not None
            and self.selected_option not in self.options
        ):
            raise ValueError("selected_option must be one of options")
        if self.status == "accepted":
            if self.selected_option is None:
                raise ValueError("accepted decisions require a selected option")
            if self.approved_by_role is None or self.approved_at is None:
                raise ValueError("accepted decisions require approval")
            if not self.evidence_reference_ids:
                raise ValueError("accepted decisions require evidence")
        return self


class EvidenceReference(BaseGovernanceRecord):
    """A provenance-bearing reference that supports or challenges a record."""

    record_type: Literal["evidence_reference"] = "evidence_reference"
    status: Literal["unverified", "verified", "failed", "blocked", "superseded"]
    evidence_kind: Literal[
        "source",
        "derivation",
        "test",
        "benchmark",
        "run",
        "artifact",
        "review",
        "external_verification",
    ]
    locator_kind: Literal[
        "local_path",
        "url",
        "doi",
        "github_run",
        "commit",
        "pull_request",
        "issue",
        "release",
    ]
    locator: str = Field(min_length=1)
    observed_at: datetime
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    supports: tuple[str, ...] = ()
    challenges: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()

    @field_validator("observed_at")
    @classmethod
    def _observation_timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_locator(self) -> EvidenceReference:
        if self.locator_kind == "local_path":
            path = PurePosixPath(self.locator.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("local evidence locators must be repository-relative")
        return self


class IssueLink(BaseGovernanceRecord):
    """Desired GitHub issue and Project projection for governance records."""

    record_type: Literal["issue_link"] = "issue_link"
    status: Literal["planned", "linked", "closed", "blocked"]
    target_record_ids: tuple[str, ...] = Field(min_length=1)
    github_repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    issue_number: int | None = Field(default=None, ge=1)
    desired_state: Literal["open", "closed"] = "open"
    managed_labels: tuple[str, ...] = ()
    project_number: int | None = Field(default=None, ge=1)
    preserve_human_content: Literal[True] = True
    close_requires_approval: Literal[True] = True


GovernanceRecord = Annotated[
    Concern | Assumption | Risk | Decision | EvidenceReference | IssueLink,
    Field(discriminator="record_type"),
]


def _record_references(record: GovernanceRecord) -> tuple[str, ...]:
    """Return every governance identifier referenced by one record."""
    references = [relation.target_id for relation in record.relations]
    references.extend(record.evidence_reference_ids)
    references.extend(record.issue_link_ids)
    if isinstance(record, EvidenceReference):
        references.extend(record.supports)
        references.extend(record.challenges)
    if isinstance(record, IssueLink):
        references.extend(record.target_record_ids)
    return tuple(references)


def _validate_typed_references(
    record: GovernanceRecord, by_id: dict[str, GovernanceRecord]
) -> None:
    """Ensure specially typed reference collections point to matching records."""
    for reference in record.evidence_reference_ids:
        if not isinstance(by_id[reference], EvidenceReference):
            raise ValueError(f"{reference} is not an evidence reference")
    for reference in record.issue_link_ids:
        if not isinstance(by_id[reference], IssueLink):
            raise ValueError(f"{reference} is not an issue link")


class GovernanceLedger(GovernanceModel):
    """Canonical collection of related governance records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        title="VOP-VOIAGE governance ledger",
    )

    schema_version: Literal["1.0.0"] = "1.0.0"
    records: tuple[GovernanceRecord, ...]

    @model_validator(mode="after")
    def _validate_graph(self) -> GovernanceLedger:
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate governance record id")
        by_id = {record.id: record for record in self.records}
        for record in self.records:
            for reference in _record_references(record):
                if reference not in by_id:
                    raise ValueError(
                        f"unknown governance record {reference} referenced by {record.id}"
                    )
            _validate_typed_references(record, by_id)
        return self


class GitHubSyncPayload(GovernanceModel):
    """Pure-data desired GitHub projection; applying it is out of scope."""

    github_repository: str
    issue_number: int | None
    desired_state: Literal["open", "closed"]
    stable_marker: str
    title: str
    body: str
    labels: tuple[str, ...]
    project_number: int | None
    project_fields: tuple[tuple[str, str], ...]


def _public_evidence_lines(
    record: BaseGovernanceRecord, by_id: dict[str, GovernanceRecord]
) -> tuple[str, ...]:
    lines: list[str] = []
    for reference in record.evidence_reference_ids:
        evidence = by_id[reference]
        if (
            isinstance(evidence, EvidenceReference)
            and evidence.visibility != "local_private"
        ):
            lines.append(f"- `{evidence.id}` — {evidence.title} ({evidence.status})")
    return tuple(lines)


def _project_fields(record: GovernanceRecord) -> tuple[tuple[str, str], ...]:
    fields: list[tuple[str, str]] = [
        ("Record ID", record.id),
        ("Record Type", record.record_type.replace("_", " ").title()),
        ("Track ID", ", ".join(record.track_ids)),
    ]
    if isinstance(record, Risk):
        fields.append(("Risk Level", record.risk_level.title()))
    return tuple(fields)


def build_github_sync_payloads(
    ledger: GovernanceLedger,
) -> tuple[GitHubSyncPayload, ...]:
    """Build deterministic, privacy-filtered GitHub payloads without mutation."""
    by_id = {record.id: record for record in ledger.records}
    payloads: list[GitHubSyncPayload] = []
    links = sorted(
        (
            record
            for record in ledger.records
            if isinstance(record, IssueLink) and record.visibility != "local_private"
        ),
        key=lambda item: item.id,
    )
    for link in links:
        for target_id in link.target_record_ids:
            target = by_id[target_id]
            if target.visibility == "local_private":
                continue
            marker = f"vop-voiage-governance-id:{target.id}"
            evidence = _public_evidence_lines(target, by_id)
            body_lines = [
                f"<!-- {marker} -->",
                "<!-- governance:begin -->",
                f"## {target.record_type.replace('_', ' ').title()}",
                "",
                target.summary,
                "",
                f"- **Record:** `{target.id}`",
                f"- **Status:** `{target.status}`",
                f"- **Tracks:** `{', '.join(target.track_ids)}`",
                "",
                "### Evidence",
                "",
                *(evidence or ("- No public evidence references.",)),
                "<!-- governance:end -->",
            ]
            payloads.append(
                GitHubSyncPayload(
                    github_repository=link.github_repository,
                    issue_number=link.issue_number,
                    desired_state=link.desired_state,
                    stable_marker=marker,
                    title=f"[{target.id}] {target.title}",
                    body="\n".join(body_lines) + "\n",
                    labels=tuple(sorted(set(link.managed_labels))),
                    project_number=link.project_number,
                    project_fields=_project_fields(target),
                )
            )
    return tuple(payloads)


_SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("concern.schema.json", Concern),
    ("assumption.schema.json", Assumption),
    ("risk.schema.json", Risk),
    ("decision.schema.json", Decision),
    ("evidence-reference.schema.json", EvidenceReference),
    ("issue-link.schema.json", IssueLink),
    ("governance-ledger.schema.json", GovernanceLedger),
    ("github-sync-payload.schema.json", GitHubSyncPayload),
)


def export_governance_schemas(output_dir: str | Path) -> tuple[Path, ...]:
    """Export deterministic JSON Schemas for governance and sync payloads."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, model in _SCHEMA_MODELS:
        path = destination / filename
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **model.model_json_schema(),
        }
        content = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)
        path.write_text(content + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


__all__ = [
    "Assumption",
    "Concern",
    "Decision",
    "EvidenceReference",
    "GitHubSyncPayload",
    "GovernanceLedger",
    "IssueLink",
    "Relation",
    "Risk",
    "build_github_sync_payloads",
    "export_governance_schemas",
]
