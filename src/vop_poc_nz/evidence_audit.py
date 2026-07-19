"""Evidence-ledger audit utilities for vop_poc_nz.

These helpers intentionally use only the Python standard library so that they can
run early in local-agent workflows before the project environment has been fully
solved. The audit is conservative: it identifies missing provenance, stale-source
signals, missing price-year/currency fields, and societal-cost taxonomy gaps, but
it does not try to decide whether a parameter is scientifically valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import csv
import re

REQUIRED_COLUMNS: tuple[str, ...] = (
    "parameter_id",
    "case_id",
    "value",
    "unit",
    "source_citation",
    "derivation_formula",
    "included_perspectives",
    "cost_component",
)

RECOMMENDED_COLUMNS: tuple[str, ...] = (
    "distribution",
    "uncertainty_rationale",
    "price_year",
    "currency",
    "source_year",
    "source_grade",
)

SOCIETAL_COST_COMPONENTS: frozenset[str] = frozenset(
    {
        "direct_medical",
        "direct_non_medical",
        "patient_time",
        "productivity",
        "informal_care",
        "out_of_pocket",
        "caregiver_spillover",
        "whanau_spillover",
        "education",
        "housing",
        "energy",
        "justice",
        "transport",
        "environment",
        "implementation",
        "administration",
    }
)

COST_HINTS = ("cost", "price", "nzd", "$", "productivity", "budget", "resource")


@dataclass(frozen=True)
class EvidenceIssue:
    """One evidence-ledger audit finding."""

    severity: str
    code: str
    message: str
    row_number: int | None = None
    parameter_id: str | None = None
    column: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "row_number": self.row_number,
            "parameter_id": self.parameter_id,
            "column": self.column,
        }


def _blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def _normalise_component(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _looks_like_cost(row: Mapping[str, object]) -> bool:
    haystack = " ".join(str(row.get(key, "")) for key in ("parameter_id", "unit", "cost_component")).lower()
    return any(hint in haystack for hint in COST_HINTS)


def _extract_year(row: Mapping[str, object]) -> int | None:
    explicit = row.get("source_year")
    if not _blank(explicit):
        try:
            return int(str(explicit).strip())
        except ValueError:
            return None
    text = " ".join(str(row.get(key, "")) for key in ("source_citation", "source_location"))
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not matches:
        return None
    return max(int(match) for match in matches)


def read_evidence_ledger(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV evidence ledger into dictionaries."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_evidence_ledger_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    current_year: int = 2026,
    stale_after_years: int = 10,
    allowed_societal_components: Sequence[str] = tuple(SOCIETAL_COST_COMPONENTS),
) -> list[EvidenceIssue]:
    """Audit evidence-ledger rows for publication-critical omissions.

    Parameters
    ----------
    rows:
        Iterable of row dictionaries, usually from :func:`read_evidence_ledger`.
    current_year:
        Year used for stale-source heuristics.
    stale_after_years:
        Sources older than this threshold are warnings, not hard failures.
    allowed_societal_components:
        Taxonomy used to flag unclassified societal costs.
    """

    materialised = list(rows)
    issues: list[EvidenceIssue] = []
    allowed = {_normalise_component(item) for item in allowed_societal_components}

    if not materialised:
        return [
            EvidenceIssue(
                severity="error",
                code="empty_ledger",
                message="Evidence ledger has no rows.",
            )
        ]

    observed_columns = set().union(*(row.keys() for row in materialised))
    for column in REQUIRED_COLUMNS:
        if column not in observed_columns:
            issues.append(
                EvidenceIssue(
                    severity="error",
                    code="missing_required_column",
                    message=f"Required column {column!r} is absent from the evidence ledger.",
                    column=column,
                )
            )
    for column in RECOMMENDED_COLUMNS:
        if column not in observed_columns:
            issues.append(
                EvidenceIssue(
                    severity="warning",
                    code="missing_recommended_column",
                    message=f"Recommended column {column!r} is absent from the evidence ledger.",
                    column=column,
                )
            )

    for idx, row in enumerate(materialised, start=1):
        parameter_id = str(row.get("parameter_id", "")).strip() or None
        for column in REQUIRED_COLUMNS:
            if column in row and _blank(row.get(column)):
                issues.append(
                    EvidenceIssue(
                        severity="error",
                        code="missing_required_value",
                        message=f"Required value {column!r} is blank.",
                        row_number=idx,
                        parameter_id=parameter_id,
                        column=column,
                    )
                )

        if _looks_like_cost(row):
            for column in ("price_year", "currency"):
                if column in observed_columns and _blank(row.get(column)):
                    issues.append(
                        EvidenceIssue(
                            severity="warning",
                            code="cost_without_price_context",
                            message=f"Cost-like parameter is missing {column!r}.",
                            row_number=idx,
                            parameter_id=parameter_id,
                            column=column,
                        )
                    )

        component = _normalise_component(row.get("cost_component"))
        perspectives = str(row.get("included_perspectives", "")).lower()
        if "societ" in perspectives and component and component not in allowed:
            issues.append(
                EvidenceIssue(
                    severity="warning",
                    code="unclassified_societal_component",
                    message=f"Societal-perspective component {component!r} is not in the structured taxonomy.",
                    row_number=idx,
                    parameter_id=parameter_id,
                    column="cost_component",
                )
            )

        if "distribution" in observed_columns and _blank(row.get("distribution")):
            issues.append(
                EvidenceIssue(
                    severity="warning",
                    code="missing_uncertainty_distribution",
                    message="Parameter has no uncertainty distribution recorded.",
                    row_number=idx,
                    parameter_id=parameter_id,
                    column="distribution",
                )
            )

        source_year = _extract_year(row)
        if source_year is not None and current_year - source_year > stale_after_years:
            issues.append(
                EvidenceIssue(
                    severity="info",
                    code="potentially_dated_source",
                    message=f"Most recent detected source year is {source_year}; check whether newer evidence exists.",
                    row_number=idx,
                    parameter_id=parameter_id,
                    column="source_citation",
                )
            )

    return issues


def audit_evidence_ledger(path: str | Path, **kwargs: object) -> list[EvidenceIssue]:
    """Read and audit a CSV evidence ledger."""

    return audit_evidence_ledger_rows(read_evidence_ledger(path), **kwargs)


def evidence_issue_summary(issues: Iterable[EvidenceIssue]) -> dict[str, int]:
    """Summarise issues by severity."""

    summary: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        summary[issue.severity] = summary.get(issue.severity, 0) + 1
    return summary
