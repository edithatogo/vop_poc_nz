#!/usr/bin/env python3
"""Check conductor/GitHub issue-body hygiene before bulk issue creation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ISSUE_PREFIX_RE = re.compile(r"^(\d+)[_-](.+)\.md$")


@dataclass(frozen=True)
class IssueRecord:
    path: str
    number: str | None
    slug: str
    title: str


@dataclass(frozen=True)
class HygieneIssue:
    severity: str
    kind: str
    message: str
    paths: list[str]


def issue_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped:
                return stripped[:100]
    except OSError:
        pass
    return path.stem


def normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def iter_issue_files(root: Path) -> Iterable[Path]:
    issue_root = root / "issues"
    if not issue_root.exists():
        return []
    return sorted(path for path in issue_root.rglob("*.md") if path.is_file())


def collect(root: Path) -> list[IssueRecord]:
    records: list[IssueRecord] = []
    for path in iter_issue_files(root):
        match = ISSUE_PREFIX_RE.match(path.name)
        number = match.group(1) if match else None
        slug = match.group(2) if match else path.stem
        records.append(
            IssueRecord(
                path=path.relative_to(root).as_posix(),
                number=number,
                slug=slug,
                title=issue_title(path),
            )
        )
    return records


def audit(root: Path) -> dict[str, object]:
    records = collect(root)
    issues: list[HygieneIssue] = []
    by_number: dict[tuple[str, str], list[IssueRecord]] = {}
    by_title: dict[tuple[str, str], list[IssueRecord]] = {}
    for record in records:
        repo_bucket = record.path.split("/")[1] if record.path.startswith("issues/") and len(record.path.split("/")) > 1 else "unknown"
        if record.number:
            by_number.setdefault((repo_bucket, record.number), []).append(record)
        else:
            issues.append(HygieneIssue("warning", "missing_number", "issue filename does not start with a numeric prefix", [record.path]))
        by_title.setdefault((repo_bucket, normalise_title(record.title)), []).append(record)
    for (bucket, number), group in sorted(by_number.items()):
        if len(group) > 1:
            issues.append(
                HygieneIssue(
                    "error",
                    "duplicate_number",
                    f"duplicate issue number {number} in {bucket}",
                    [record.path for record in group],
                )
            )
    for (bucket, title), group in sorted(by_title.items()):
        if title and len(group) > 1:
            issues.append(
                HygieneIssue(
                    "warning",
                    "duplicate_title",
                    f"duplicate or near-duplicate issue title in {bucket}: {title}",
                    [record.path for record in group],
                )
            )
    return {
        "records": [asdict(record) for record in records],
        "issues": [asdict(issue) for issue in issues],
        "error_count": sum(1 for issue in issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
    }


def to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Issue hygiene audit",
        "",
        f"- Issue files: {len(report.get('records', []))}",
        f"- Errors: {report.get('error_count', 0)}",
        f"- Warnings: {report.get('warning_count', 0)}",
        "",
    ]
    issues = report.get("issues", [])
    if issues:
        lines.append("| Severity | Kind | Message | Paths |")
        lines.append("|---|---|---|---|")
        for issue in issues:  # type: ignore[assignment]
            paths = "<br>".join(f"`{path}`" for path in issue["paths"])
            message = str(issue["message"]).replace("|", "\\|")
            lines.append(f"| {issue['severity']} | {issue['kind']} | {message} | {paths} |")
    else:
        lines.append("No issue hygiene problems detected.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    report = audit(root)
    local_dir = root / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "issue_hygiene.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (local_dir / "issue_hygiene.md").write_text(to_markdown(report), encoding="utf-8")
    print(local_dir / "issue_hygiene.md")
    if args.strict and report.get("error_count", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
