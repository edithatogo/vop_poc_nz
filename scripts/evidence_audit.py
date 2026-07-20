#!/usr/bin/env python3
"""Audit evidence ledgers and parameter provenance before publication.

The audit is schema-light and dependency-free so it can run before the project
environment is installed. It supports CSV ledgers and simple JSON arrays. The
purpose is not to validate every citation automatically; it is to stop manuscript
or repository outputs from claiming empirical support when key provenance fields
are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_FIELDS = {
    "parameter_id",
    "case_id",
    "value",
    "unit",
    "source_citation",
    "derivation_formula",
    "uncertainty_rationale",
}
RECOMMENDED_FIELDS = {
    "distribution",
    "price_year",
    "currency",
    "perspective",
    "cost_component",
    "source_location",
    "source_quality",
    "included_in_base_case",
}
LEDGER_PATTERNS = ("**/*evidence*ledger*.csv", "**/evidence_ledger*.csv", "**/*evidence*ledger*.json")
SKIP_PARTS = {".git", ".venv", ".pixi", "node_modules", ".conductor"}


@dataclass(frozen=True)
class LedgerIssue:
    ledger: str
    row: int
    severity: str
    field: str
    message: str
    parameter_id: str | None = None


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def find_ledgers(root: Path) -> list[Path]:
    ledgers: list[Path] = []
    seen: set[Path] = set()
    for pattern in LEDGER_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
                continue
            if path not in seen:
                ledgers.append(path)
                seen.add(path)
    return sorted(ledgers)


def read_rows(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader], list(reader.fieldnames or [])
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        records = data["records"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("JSON evidence ledger must be a list or object with records[]")
    rows = [dict(item) for item in records if isinstance(item, dict)]
    fields = sorted({key for row in rows for key in row})
    return rows, fields


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "NA", "N/A", "null", "None", "?"}:
        return True
    return False


def audit_ledger(path: Path, root: Path) -> tuple[dict[str, object], list[LedgerIssue]]:
    rel = rel_posix(path, root)
    issues: list[LedgerIssue] = []
    try:
        rows, fields = read_rows(path)
    except Exception as exc:  # pragma: no cover - exact errors vary
        issues.append(LedgerIssue(rel, 0, "error", "file", f"Could not parse ledger: {exc}", None))
        return {"path": rel, "rows": 0, "fields": [], "parse_status": "error"}, issues

    field_set = set(fields)
    for field in sorted(REQUIRED_FIELDS - field_set):
        issues.append(LedgerIssue(rel, 0, "error", field, "Required column is missing", None))
    for field in sorted(RECOMMENDED_FIELDS - field_set):
        issues.append(LedgerIssue(rel, 0, "warning", field, "Recommended column is missing", None))

    for index, row in enumerate(rows, start=1):
        parameter_id = str(row.get("parameter_id") or "").strip() or None
        for field in sorted(REQUIRED_FIELDS & field_set):
            if is_missing(row.get(field)):
                issues.append(LedgerIssue(rel, index, "error", field, "Required value is missing", parameter_id))
        for field in sorted(RECOMMENDED_FIELDS & field_set):
            if is_missing(row.get(field)):
                issues.append(LedgerIssue(rel, index, "warning", field, "Recommended value is missing", parameter_id))
        if not is_missing(row.get("value")) and not is_missing(row.get("distribution")):
            distribution = str(row.get("distribution", "")).strip().lower()
            if distribution in {"fixed", "point", "deterministic"} and is_missing(row.get("uncertainty_rationale")):
                issues.append(
                    LedgerIssue(rel, index, "warning", "uncertainty_rationale", "Fixed parameter needs uncertainty rationale", parameter_id)
                )
    return {"path": rel, "rows": len(rows), "fields": fields, "parse_status": "ok"}, issues


def build_audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    ledgers = find_ledgers(root)
    ledger_summaries: list[dict[str, object]] = []
    issues: list[LedgerIssue] = []
    for path in ledgers:
        summary, found = audit_ledger(path, root)
        ledger_summaries.append(summary)
        issues.extend(found)
    counts = Counter(issue.severity for issue in issues)
    recommendations: list[str] = []
    if not ledgers:
        recommendations.append("No evidence ledger found; create one before describing inputs as empirically sourced.")
    if counts.get("error", 0):
        recommendations.append("Resolve missing required provenance fields before journal/preprint release.")
    if counts.get("warning", 0):
        recommendations.append("Review recommended provenance fields so readers can trace price year, currency, perspective, and uncertainty assumptions.")
    return {
        "schema_version": "1.0",
        "repo": {"root": str(root), "name": root.name},
        "summary": {
            "ledgers_found": len(ledgers),
            "issues_total": len(issues),
            "severity_counts": dict(counts),
            "ready_for_empirical_claims": bool(ledgers) and counts.get("error", 0) == 0,
        },
        "ledgers": ledger_summaries,
        "issues": [asdict(issue) for issue in issues],
        "recommendations": recommendations,
    }


def to_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [f"# Evidence ledger audit: {report['repo']['name']}", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    if report["ledgers"]:
        lines.extend(["", "## Ledgers", "", "| Path | Rows | Fields | Status |", "|---|---:|---|---|"])
        for ledger in report["ledgers"]:
            lines.append(f"| `{ledger['path']}` | {ledger['rows']} | {len(ledger['fields'])} | `{ledger['parse_status']}` |")
    if report["issues"]:
        lines.extend(["", "## Issues", "", "| Severity | Ledger | Row | Field | Parameter | Message |", "|---|---|---:|---|---|---|"])
        for issue in report["issues"][:300]:
            lines.append(
                f"| `{issue['severity']}` | `{issue['ledger']}` | {issue['row']} | `{issue['field']}` | `{issue['parameter_id']}` | {issue['message']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = build_audit(args.repo)
    out_dir = args.repo / ".conductor" / "local"
    output_json = args.output_json or out_dir / "evidence_audit.json"
    output_md = args.output_md or out_dir / "evidence_audit.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Evidence audit written: {output_md}")
    if args.strict and not report["summary"]["ready_for_empirical_claims"]:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
