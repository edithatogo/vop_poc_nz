"""Claim-boundary linting for manuscripts and reports.

This checker is intentionally conservative. It does not ban claims; it forces
claims that reviewers are likely to challenge to be backed by an explicit marker,
such as a citation, model-card reference, evidence-ledger reference, or softer
wording. It is designed for pre-commit/CI use on Markdown, Quarto, or LaTeX text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Pattern


@dataclass(frozen=True)
class ClaimRule:
    """One claim-boundary rule."""

    rule_id: str
    pattern: Pattern[str]
    message: str
    allowed_markers: tuple[str, ...] = (
        "\\cite",
        "[@",
        "evidence_ledger",
        "case_contract",
        "model_card",
        "assumption_ledger",
        "policy-grade: true",
        "decision-grade: true",
    )


@dataclass(frozen=True)
class ClaimFinding:
    """A claim-boundary finding."""

    rule_id: str
    line_number: int
    line: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "line_number": self.line_number,
            "line": self.line,
            "message": self.message,
        }


DEFAULT_RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        "empirical-demonstration",
        re.compile(r"\bempirical\s+demonstration\b", re.IGNORECASE),
        "Use 'empirically parameterised simulation' unless this is a de novo empirical estimate.",
    ),
    ClaimRule(
        "policy-grade",
        re.compile(r"\b(policy[- ]grade|decision[- ]grade|validated model)\b", re.IGNORECASE),
        "Policy/decision-grade or validated claims require a model card or validation record.",
    ),
    ClaimRule(
        "dominance",
        re.compile(r"\b(dominant|dominance|cost[- ]saving)\b", re.IGNORECASE),
        "Dominance/cost-saving claims require threshold, cost-component, and uncertainty context.",
    ),
    ClaimRule(
        "first-novel",
        re.compile(r"\b(first|novel|unique|only)\b", re.IGNORECASE),
        "Priority/novelty claims require a software/literature comparison reference.",
    ),
    ClaimRule(
        "societal-bonus",
        re.compile(r"\bsocietal\s+bonus\b", re.IGNORECASE),
        "Prefer directional perspective regret/EVoP language; 'societal bonus' is colloquial.",
        allowed_markers=("EVoP", "perspective regret", "assumption_ledger", "evidence_ledger"),
    ),
)


def _has_allowed_marker(line: str, rule: ClaimRule) -> bool:
    lowered = line.lower()
    return any(marker.lower() in lowered for marker in rule.allowed_markers)


def lint_text(text: str, *, rules: Iterable[ClaimRule] = DEFAULT_RULES) -> list[ClaimFinding]:
    """Return claim-boundary findings for one text document."""
    findings: list[ClaimFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("# noqa: claim-boundary"):
            continue
        for rule in rules:
            if rule.pattern.search(line) and not _has_allowed_marker(line, rule):
                findings.append(
                    ClaimFinding(
                        rule_id=rule.rule_id,
                        line_number=line_number,
                        line=line.rstrip(),
                        message=rule.message,
                    )
                )
    return findings


def lint_file(path: str | Path, *, rules: Iterable[ClaimRule] = DEFAULT_RULES) -> list[ClaimFinding]:
    """Return findings for a text file."""
    return lint_text(Path(path).read_text(encoding="utf-8"), rules=rules)


def format_findings(findings: Iterable[ClaimFinding]) -> str:
    """Format findings for CLI/CI logs."""
    return "\n".join(
        f"{finding.line_number}: {finding.rule_id}: {finding.message}\n"
        f"    {finding.line}"
        for finding in findings
    )
