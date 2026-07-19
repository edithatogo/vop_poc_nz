"""Contract tests for privacy-safe concern governance."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vop_poc_nz.concerns import (
    Assumption,
    Concern,
    Decision,
    EvidenceReference,
    GovernanceLedger,
    IssueLink,
    Risk,
    build_github_sync_payloads,
    export_governance_schemas,
)

NOW = datetime(2026, 7, 19, tzinfo=UTC)


def _records() -> tuple[object, ...]:
    evidence = EvidenceReference(
        id="EVR-SHR-0001",
        title="Hosted conformance evidence",
        summary="The shared Arrow conformance suite passed.",
        status="verified",
        visibility="public",
        evidence_kind="test",
        locator_kind="github_run",
        locator="https://github.com/edithatogo/vop_poc_nz/actions/runs/1",
        observed_at=NOW,
        supports=("RSK-SHR-0001", "DEC-SHR-0001"),
    )
    private_evidence = EvidenceReference(
        id="EVR-VOP-0002",
        title="Private reviewer note",
        summary="Unpublished owner-controlled evidence.",
        status="unverified",
        visibility="local_private",
        evidence_kind="review",
        locator_kind="local_path",
        locator=".conductor/local/reviewer-note.md",
        observed_at=NOW,
        supports=("RSK-SHR-0001",),
    )
    concern = Concern(
        id="CON-SHR-0001",
        title="Interchange drift",
        summary="The two repositories may silently diverge.",
        status="investigating",
        question="How is cross-repository drift detected?",
        impact_if_unresolved="Consumers could read incompatible outputs.",
        resolution_criteria=("Pinned schema digests agree.",),
        track_ids=("C09",),
    )
    assumption = Assumption(
        id="ASM-SHR-0001",
        title="Canonical JSON stability",
        summary="Ordered logical fields provide a stable fingerprint input.",
        status="active",
        category="computational",
        rationale="The representation is language-neutral.",
        validation_method="Cross-language golden fixture comparison.",
        falsification_condition="Two conforming implementations produce different hashes.",
        review_due="2026-10-19",
        track_ids=("C09",),
    )
    risk = Risk(
        id="RSK-SHR-0001",
        title="Schema drift",
        summary="A consumer may accept an incompatible schema.",
        status="mitigating",
        cause="Independent repository evolution.",
        event="A mirrored contract changes without a version bump.",
        consequence="Interchange validation becomes unsound.",
        likelihood="possible",
        impact="major",
        risk_level="high",
        mitigations=("Pin canonical commit and digest.",),
        evidence_reference_ids=(evidence.id, private_evidence.id),
        issue_link_ids=("ISL-SHR-0001",),
        track_ids=("C09",),
    )
    decision = Decision(
        id="DEC-SHR-0001",
        title="Use a pinned canonical contract",
        summary="VOP owns the canonical contract and VOIAGE pins a mirror.",
        status="accepted",
        question="How should shared compatibility policy be governed?",
        options=("Pinned canonical mirror", "Unpinned duplicate files"),
        selected_option="Pinned canonical mirror",
        rationale="It permits offline use while making drift measurable.",
        consequences=("Contract updates require coordinated versioning.",),
        reversibility="reversible",
        approved_by_role="owner",
        approved_at=NOW,
        evidence_reference_ids=(evidence.id,),
        track_ids=("C09",),
    )
    link = IssueLink(
        id="ISL-SHR-0001",
        title="Track schema drift risk",
        summary="Project coordination link for the shared risk.",
        status="linked",
        target_record_ids=(risk.id,),
        github_repository="edithatogo/vop_poc_nz",
        issue_number=41,
        desired_state="open",
        managed_labels=("gov:risk", "track:C09"),
        project_number=28,
        track_ids=("C12",),
    )
    return evidence, private_evidence, concern, assumption, risk, decision, link


def test_models_are_strict_frozen_and_forbid_extra_fields() -> None:
    concern = _records()[2]
    with pytest.raises(ValidationError):
        Concern.model_validate({**concern.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        Concern(
            id="CON-SHR-0002",
            title="Invalid coercion",
            summary="Strict models reject coercion.",
            status="open",
            question="Will an integer be coerced?",
            impact_if_unresolved="The contract would be ambiguous.",
            resolution_criteria="not-a-tuple",
        )
    with pytest.raises(ValidationError):
        concern.title = "mutated"  # type: ignore[misc]


def test_ledger_validates_unique_ids_and_references() -> None:
    ledger = GovernanceLedger(records=_records())
    assert len(ledger.records) == 7
    with pytest.raises(ValidationError, match="duplicate governance record id"):
        GovernanceLedger(records=(*_records(), _records()[0]))
    orphan = Risk(
        id="RSK-SHR-0099",
        title="Orphan",
        summary="References missing evidence.",
        status="open",
        cause="Missing record.",
        event="Reference cannot resolve.",
        consequence="Traceability is incomplete.",
        likelihood="unlikely",
        impact="minor",
        risk_level="low",
        evidence_reference_ids=("EVR-SHR-9999",),
    )
    with pytest.raises(ValidationError, match="unknown governance record"):
        GovernanceLedger(records=(orphan,))


def test_github_projection_is_deterministic_and_privacy_safe() -> None:
    ledger = GovernanceLedger(records=_records())
    first = build_github_sync_payloads(ledger)
    second = build_github_sync_payloads(ledger)

    assert first == second
    assert len(first) == 1
    payload = first[0]
    assert payload.stable_marker == "vop-voiage-governance-id:RSK-SHR-0001"
    assert "Hosted conformance evidence" in payload.body
    assert "Private reviewer note" not in payload.body
    assert ".conductor/local" not in payload.body
    assert payload.project_fields == (
        ("Record ID", "RSK-SHR-0001"),
        ("Record Type", "Risk"),
        ("Track ID", "C09"),
        ("Risk Level", "High"),
    )


def test_schema_export_is_deterministic(tmp_path) -> None:
    first = export_governance_schemas(tmp_path)
    before = {path.name: path.read_bytes() for path in first}
    second = export_governance_schemas(tmp_path)

    assert first == second
    assert before == {path.name: path.read_bytes() for path in second}
    ledger_schema = json.loads((tmp_path / "governance-ledger.schema.json").read_text())
    assert ledger_schema["title"] == "VOP-VOIAGE governance ledger"
    assert ledger_schema["additionalProperties"] is False
    for path in first:
        schema = json.loads(path.read_text())
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
