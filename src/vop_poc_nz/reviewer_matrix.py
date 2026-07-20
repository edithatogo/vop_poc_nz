"""Reviewer-response matrix helpers for the vop_poc_nz manuscript family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReviewerConcern:
    id: str
    source: str
    concern: str
    required_action: str
    owner_track: str
    manuscript_target: str

    def as_markdown_row(self) -> str:
        cells = [self.id, self.source, self.concern, self.required_action, self.owner_track, self.manuscript_target]
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


CORE_REVIEWER_CONCERNS: tuple[ReviewerConcern, ...] = (
    ReviewerConcern(
        id="R-SCOPE-001",
        source="JRSNZ/R2",
        concern="Manuscript attempts multiple interventions and frameworks at the expense of depth.",
        required_action="Separate one deep exemplar from tutorial/comparative case suite; enforce concept scope budget.",
        owner_track="track_27_concept_scope_budget.md",
        manuscript_target="preprint_v2; NZ application paper",
    ),
    ReviewerConcern(
        id="R-PARAM-002",
        source="JRSNZ/R2",
        concern="Key assumptions and parameter sources are insufficiently documented.",
        required_action="Complete evidence ledgers, case contracts, and main-text assumption ledger.",
        owner_track="track_28_evidence_audit_and_source_refresh.md",
        manuscript_target="preprint_v2; NZ application paper",
    ),
    ReviewerConcern(
        id="R-MODEL-003",
        source="JRSNZ/R2",
        concern="Three-state Markov models need clinical and epidemiological justification.",
        required_action="Add model cards with structure rationale and decision-grade status for each case.",
        owner_track="track_09_case_contracts_evidence_model_cards.md",
        manuscript_target="preprint_v2 supplement; NZ application paper main text",
    ),
    ReviewerConcern(
        id="R-SOC-004",
        source="JRSNZ/R2",
        concern="Societal perspective is treated largely as productivity rather than a structured cost framework.",
        required_action="Add societal-cost taxonomy and inclusion/exclusion table per case.",
        owner_track="track_10_societal_cost_taxonomy.md",
        manuscript_target="preprint_v2 methods; NZ application paper",
    ),
    ReviewerConcern(
        id="R-VOI-005",
        source="NZMJ/R1; JRSNZ/R1",
        concern="VOI/EVPPI purpose and policy implications need clearer explanation.",
        required_action="Keep VOI as supporting analysis; explain relationship between EVPI and directional EVoP.",
        owner_track="track_11_cheers_voi_submission_package.md",
        manuscript_target="preprint_v2; Value in Health methods paper",
    ),
    ReviewerConcern(
        id="R-HARM-006",
        source="JRSNZ/R1",
        concern="Asymmetric adoption of broad societal perspective across sectors could distort public investment.",
        required_action="Treat Value of Harmonisation as scoped discussion/extension, not a new headline result.",
        owner_track="track_12_value_of_harmonisation_extension.md",
        manuscript_target="NZ application paper discussion",
    ),
    ReviewerConcern(
        id="R-NOVEL-007",
        source="JRSNZ/R2; NZMJ editor",
        concern="Including productivity benefits is expected and not sufficiently novel by itself.",
        required_action="Reframe novelty around directional EVoP, Perspective Acceptability Frontier, and regime discovery.",
        owner_track="track_29_paf_mixture_frontier_implementation.md",
        manuscript_target="Value in Health methods paper; preprint_v2",
    ),
)


def reviewer_response_matrix(concerns: Iterable[ReviewerConcern] = CORE_REVIEWER_CONCERNS) -> str:
    lines = [
        "# Reviewer-response matrix",
        "",
        "| ID | Source | Concern | Required action | Owner track | Manuscript target |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(concern.as_markdown_row() for concern in concerns)
    return "\n".join(lines) + "\n"
