#!/usr/bin/env python3
"""Validate the canonical conductor and issue registries.

The registry is intentionally JSON-only so it can run before project dependencies
are installed. It detects duplicate IDs, missing files, invalid dependencies,
cycles, and issue-to-track mismatches.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

VALID_STATUSES = {"planned", "ready", "in_progress", "blocked", "completed", "superseded"}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing registry: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Registry root must be an object: {path}")
    return data


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for value in values:
        if value in seen:
            dup.add(value)
        seen.add(value)
    return dup


def _cycle_nodes(graph: dict[str, list[str]]) -> set[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            if node in stack:
                cycles.update(stack[stack.index(node) :])
            else:
                cycles.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for dep in graph.get(node, []):
            if dep in graph:
                visit(dep, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, [])
    return cycles


def validate(pack_root: Path) -> dict[str, Any]:
    pack_root = pack_root.resolve()
    manifest_path = pack_root / "conductor" / "manifest.json"
    backlog_path = pack_root / "issues" / "backlog.json"
    findings: list[Finding] = []

    manifest = load_json(manifest_path)
    tracks = manifest.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError("conductor.manifest tracks must be a list")

    track_ids = [str(track.get("id", "")) for track in tracks if isinstance(track, dict)]
    track_slugs = [str(track.get("slug", "")) for track in tracks if isinstance(track, dict) and track.get("slug")]
    track_files = [str(track.get("file", "")) for track in tracks if isinstance(track, dict) and track.get("file")]
    for duplicate in sorted(_duplicates(track_ids)):
        findings.append(Finding("error", "duplicate_track_id", f"Duplicate track ID: {duplicate}"))
    for duplicate in sorted(_duplicates(track_slugs)):
        findings.append(Finding("error", "duplicate_track_slug", f"Duplicate track slug: {duplicate}"))
    for duplicate in sorted(_duplicates(track_files)):
        findings.append(Finding("error", "duplicate_track_file", f"Duplicate track file: {duplicate}"))
    known_tracks = set(track_ids)
    graph: dict[str, list[str]] = {}
    for track in tracks:
        if not isinstance(track, dict):
            findings.append(Finding("error", "invalid_track", "Track entry is not an object"))
            continue
        track_id = str(track.get("id", ""))
        file_value = str(track.get("file", ""))
        if not track_id:
            findings.append(Finding("error", "missing_track_id", "Track has no ID"))
        if file_value and not (pack_root / "conductor" / file_value).exists():
            findings.append(Finding("error", "missing_track_file", f"{track_id}: missing {file_value}"))
        status = str(track.get("default_status", "planned"))
        if status not in VALID_STATUSES:
            findings.append(Finding("error", "invalid_track_status", f"{track_id}: {status}"))
        deps = [str(dep) for dep in track.get("depends_on", [])]
        graph.setdefault(track_id, [])
        graph[track_id].extend(dep for dep in deps if dep not in graph[track_id])
        for dep in deps:
            if dep not in known_tracks:
                findings.append(Finding("error", "missing_dependency", f"{track_id} depends on unknown {dep}"))
    cycles = _cycle_nodes(graph)
    if cycles:
        findings.append(Finding("error", "dependency_cycle", f"Track dependency cycle includes: {', '.join(sorted(cycles))}"))

    backlog = load_json(backlog_path)
    issues = backlog.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("issues.backlog issues must be a list")
    issue_ids = [str(issue.get("id", "")) for issue in issues if isinstance(issue, dict)]
    titles = [str(issue.get("title", "")).strip().casefold() for issue in issues if isinstance(issue, dict) and str(issue.get("title", "")).strip()]
    for duplicate in sorted(_duplicates(issue_ids)):
        findings.append(Finding("error", "duplicate_issue_id", f"Duplicate issue ID: {duplicate}"))
    for duplicate in sorted(_duplicates(titles)):
        findings.append(Finding("error", "duplicate_issue_title", f"Duplicate issue title: {duplicate}"))
    known_issues = set(issue_ids)
    issue_graph: dict[str, list[str]] = {}
    for issue in issues:
        if not isinstance(issue, dict):
            findings.append(Finding("error", "invalid_issue", "Issue entry is not an object"))
            continue
        issue_id = str(issue.get("id", ""))
        track = str(issue.get("track", ""))
        repository = str(issue.get("repository", ""))
        if repository not in {"vop_poc_nz", "voiage"}:
            findings.append(Finding("error", "invalid_issue_repository", f"{issue_id} has invalid repository {repository!r}"))
        if track not in known_tracks:
            findings.append(Finding("error", "unknown_issue_track", f"{issue_id} references unknown track {track}"))
        deps = [str(dep) for dep in issue.get("depends_on", [])]
        issue_graph[issue_id] = deps
        for dep in deps:
            if dep not in known_issues:
                findings.append(Finding("error", "unknown_issue_dependency", f"{issue_id} depends on unknown issue {dep}"))
    issue_cycles = _cycle_nodes(issue_graph)
    if issue_cycles:
        findings.append(Finding("error", "issue_dependency_cycle", f"Issue dependency cycle includes: {', '.join(sorted(issue_cycles))}"))

    summary = {
        "track_count": len(tracks),
        "issue_count": len(issues),
        "errors": sum(item.severity == "error" for item in findings),
        "warnings": sum(item.severity == "warning" for item in findings),
        "valid": not any(item.severity == "error" for item in findings),
    }
    return {
        "schema_version": "1.0",
        "pack_root": str(pack_root),
        "summary": summary,
        "findings": [asdict(item) for item in findings],
    }


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Conductor registry validation",
        "",
        f"- Tracks: **{summary['track_count']}**",
        f"- Issues: **{summary['issue_count']}**",
        f"- Errors: **{summary['errors']}**",
        f"- Warnings: **{summary['warnings']}**",
        f"- Valid: **{summary['valid']}**",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        for item in report["findings"]:
            lines.append(f"- **{item['severity'].upper()} `{item['code']}`:** {item['message']}")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    report = validate(args.pack_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report), end="")
    return 0 if report["summary"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
