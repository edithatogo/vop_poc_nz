#!/usr/bin/env python3
"""Audit evidence ledgers for parameter transparency and publication readiness."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = {
    "parameter_id",
    "case_id",
    "value",
    "unit",
    "source_citation",
    "source_grade",
    "derivation_formula",
    "included_perspectives",
}

RECOMMENDED_FIELDS = {
    "distribution",
    "price_year",
    "currency",
    "uncertainty_rationale",
    "cost_component",
    "source_location",
}

ALLOWED_SOURCE_GRADES = {
    "empirical_published",
    "empirical_administrative",
    "expert_elicitation",
    "calibration",
    "assumption",
    "synthetic_fixture",
}


@dataclass(frozen=True)
class LedgerIssue:
    file: str
    row: int
    severity: str
    field: str
    message: str


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        if isinstance(data.get("parameters"), list):
            return [dict(item) for item in data["parameters"]]
        return [data]
    raise ValueError(f"Unsupported JSON evidence ledger structure: {path}")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".json":
        return read_json(path)
    raise ValueError("Only CSV and JSON evidence ledgers are audited without optional dependencies")


def normalise_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def audit_rows(path: Path, rows: list[dict[str, Any]]) -> list[LedgerIssue]:
    issues: list[LedgerIssue] = []
    if not rows:
        return [LedgerIssue(str(path), 0, "error", "<file>", "ledger contains no parameter rows")]
    fields = set(rows[0].keys())
    for field in sorted(REQUIRED_FIELDS - fields):
        issues.append(LedgerIssue(str(path), 0, "error", field, "required field is missing from ledger"))
    for field in sorted(RECOMMENDED_FIELDS - fields):
        issues.append(LedgerIssue(str(path), 0, "warning", field, "recommended field is missing from ledger"))
    for idx, row in enumerate(rows, start=2):
        for field in sorted(REQUIRED_FIELDS & set(row.keys())):
            if normalise_cell(row.get(field)) == "":
                issues.append(LedgerIssue(str(path), idx, "error", field, "required value is blank"))
        source_grade = normalise_cell(row.get("source_grade"))
        if source_grade and source_grade not in ALLOWED_SOURCE_GRADES:
            issues.append(
                LedgerIssue(
                    str(path),
                    idx,
                    "warning",
                    "source_grade",
                    f"unexpected source grade {source_grade!r}; use a controlled vocabulary or justify it",
                )
            )
        price_year = normalise_cell(row.get("price_year"))
        if price_year:
            try:
                year = int(float(price_year))
            except ValueError:
                issues.append(LedgerIssue(str(path), idx, "warning", "price_year", "price year is not numeric"))
            else:
                if year < 2015:
                    issues.append(
                        LedgerIssue(str(path), idx, "warning", "price_year", "price year looks old; document inflation adjustment")
                    )
        perspectives = normalise_cell(row.get("included_perspectives"))
        if perspectives and "societal" in perspectives.lower() and not normalise_cell(row.get("cost_component")):
            issues.append(
                LedgerIssue(
                    str(path),
                    idx,
                    "warning",
                    "cost_component",
                    "societal-perspective parameter should identify a societal-cost component",
                )
            )
    return issues


def discover_ledgers(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in (
        "examples/*evidence*ledger*.csv",
        "examples/*evidence*ledger*.json",
        "data/**/*evidence*ledger*.csv",
        "data/**/*evidence*ledger*.json",
        "manuscripts/**/*evidence*ledger*.csv",
        "manuscripts/**/*evidence*ledger*.json",
    ):
        candidates.extend(root.glob(pattern))
    return sorted({path for path in candidates if path.is_file()})


def audit_files(paths: Iterable[Path]) -> dict[str, Any]:
    file_reports: list[dict[str, Any]] = []
    all_issues: list[LedgerIssue] = []
    for path in paths:
        try:
            rows = read_ledger(path)
        except Exception as exc:  # noqa: BLE001 - audit should report bad files, not crash silently
            issues = [LedgerIssue(str(path), 0, "error", "<file>", str(exc))]
            rows = []
        else:
            issues = audit_rows(path, rows)
        all_issues.extend(issues)
        file_reports.append({"file": str(path), "rows": len(rows), "issues": [asdict(issue) for issue in issues]})
    return {
        "files": file_reports,
        "issue_count": len(all_issues),
        "error_count": sum(1 for issue in all_issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in all_issues if issue.severity == "warning"),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence ledger audit",
        "",
        f"- Files audited: {len(report.get('files', []))}",
        f"- Errors: {report.get('error_count', 0)}",
        f"- Warnings: {report.get('warning_count', 0)}",
        "",
    ]
    for file_report in report.get("files", []):
        lines.extend([f"## `{file_report['file']}`", "", f"Rows: {file_report['rows']}", ""])
        issues = file_report.get("issues", [])
        if not issues:
            lines.append("No issues detected.\n")
            continue
        lines.append("| Row | Severity | Field | Message |")
        lines.append("|---:|---|---|---|")
        for issue in issues:
            message = str(issue["message"]).replace("|", "\\|")
            lines.append(f"| {issue['row']} | {issue['severity']} | `{issue['field']}` | {message} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="Repository root")
    parser.add_argument("--ledger", action="append", type=Path, default=[], help="Specific ledger to audit")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    ledgers = [path.resolve() for path in args.ledger] or discover_ledgers(root)
    report = audit_files(ledgers)
    local_dir = root / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "evidence_ledger_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (local_dir / "evidence_ledger_audit.md").write_text(to_markdown(report), encoding="utf-8")
    print(local_dir / "evidence_ledger_audit.md")
    if args.strict and report.get("error_count", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
