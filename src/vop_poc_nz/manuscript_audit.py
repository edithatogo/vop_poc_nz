"""Manuscript-scope and terminology audit for the vop_poc_nz preprint.

The goal is to prevent the revised arXiv manuscript from repeating the same
scope, terminology, and traceability problems identified by reviewers. This is a
lightweight static audit; it complements, rather than replaces, human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

HEADLINE_CONCEPTS: tuple[str, ...] = (
    "directional EVoP",
    "Perspective Acceptability Frontier",
    "perspective regime discovery",
)

SUPPORTING_CONCEPTS: tuple[str, ...] = (
    "CEA",
    "DCEA",
    "BIA",
    "EVPI",
    "EVPPI",
    "MCDA",
    "Value of Harmonisation",
    "policy brief",
)

TERM_REPLACEMENTS: dict[str, str] = {
    "societal bonus": "Use 'perspective gap' for ΔNMB or 'EVoP' only for directional regret.",
    "true welfare standard": "Avoid implying that the societal perspective is automatically authoritative; specify the target perspective.",
    "wrong funding decision": "Use 'discordant decision under the target perspective' unless a normative authority is explicit.",
    "empirical demonstration": "Use 'empirically parameterised simulation' unless the analysis is a de novo empirical estimate.",
    "hypothetical interventions": "Use 'empirically parameterised case studies' only when the evidence ledger supports it.",
    "first open-source": "Use only if the software comparison and claim ledger support priority.",
}

REQUIRED_METHOD_TERMS: tuple[str, ...] = (
    "directional",
    "regret",
    "choose_under",
    "evaluate_under",
)


@dataclass(frozen=True)
class ManuscriptIssue:
    severity: str
    code: str
    message: str
    location: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location,
        }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _headings(text: str) -> list[str]:
    markdown = re.findall(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE)
    latex = re.findall(r"\\(?:section|subsection|subsubsection)\{([^}]+)\}", text)
    return markdown + latex


def audit_manuscript_text(text: str, *, max_supporting_headline_sections: int = 3) -> list[ManuscriptIssue]:
    """Audit manuscript text for over-scope and ambiguous VoP terminology."""

    issues: list[ManuscriptIssue] = []
    lower = text.lower()
    headings = _headings(text)

    for term, replacement in TERM_REPLACEMENTS.items():
        if term.lower() in lower:
            issues.append(
                ManuscriptIssue(
                    severity="warning",
                    code="ambiguous_or_overclaiming_term",
                    message=f"Found {term!r}. {replacement}",
                )
            )

    if "value of perspective" in lower:
        missing = [term for term in REQUIRED_METHOD_TERMS if term.lower() not in lower]
        if missing:
            issues.append(
                ManuscriptIssue(
                    severity="error",
                    code="evop_semantics_underdefined",
                    message="Value of Perspective is discussed but these implementation semantics are missing: "
                    + ", ".join(missing),
                    location="methods",
                )
            )

    supporting_in_headings = [
        heading
        for heading in headings
        for concept in SUPPORTING_CONCEPTS
        if concept.lower() in heading.lower()
    ]
    if len(supporting_in_headings) > max_supporting_headline_sections:
        issues.append(
            ManuscriptIssue(
                severity="warning",
                code="concept_scope_budget_exceeded",
                message=(
                    "Supporting concepts appear as major headings too often. Keep the manuscript centred on "
                    + ", ".join(HEADLINE_CONCEPTS)
                    + f". Supporting headings detected: {supporting_in_headings}"
                ),
                location="headings",
            )
        )

    if "perspective acceptability frontier" not in lower and "value in health" in lower:
        issues.append(
            ManuscriptIssue(
                severity="warning",
                code="missing_paf_for_methods_paper",
                message="A Value in Health methods manuscript should foreground the Perspective Acceptability Frontier.",
                location="methods/results",
            )
        )

    for acronym in ("CHEERS", "VOI", "EVPI", "EVPPI"):
        if acronym.lower() in lower and "checklist" not in lower:
            issues.append(
                ManuscriptIssue(
                    severity="info",
                    code="checklist_or_definition_needed",
                    message=f"{acronym} appears but no checklist/definition signal was detected.",
                )
            )

    if re.search(r"\bdominant\b", lower) and "dominant means" not in lower and "dominance" not in lower:
        issues.append(
            ManuscriptIssue(
                severity="info",
                code="dominance_definition_needed",
                message="Define dominance at first use for non-economist readers.",
            )
        )

    return issues


def audit_manuscript_file(path: str | Path, **kwargs: object) -> list[ManuscriptIssue]:
    """Audit a markdown, LaTeX, or plain-text manuscript file."""

    return audit_manuscript_text(Path(path).read_text(encoding="utf-8"), **kwargs)


def render_manuscript_audit_markdown(issues: list[ManuscriptIssue]) -> str:
    """Render audit issues as a compact markdown table."""

    lines = ["# Manuscript audit", "", "| Severity | Code | Location | Message |", "|---|---|---|---|"]
    for issue in issues:
        message = _normalise(issue.message).replace("|", "\\|")
        lines.append(f"| {issue.severity} | {issue.code} | {issue.location or ''} | {message} |")
    if not issues:
        lines.append("| ok | none |  | No issues detected. |")
    return "\n".join(lines) + "\n"
