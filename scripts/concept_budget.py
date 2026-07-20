#!/usr/bin/env python3
"""Audit manuscript/docs concept scope so the project does not over-expand."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

DEFAULT_ALLOWED_HEADLINE_CONCEPTS = {
    "expected value of perspective": [r"expected value of perspective", r"\bevop\b", r"directional evop"],
    "perspective acceptability frontier": [r"perspective acceptability frontier", r"\bpaf\b"],
    "perspective regime discovery": [r"perspective regime discovery", r"regime discovery"],
}

SUPPORTING_CONCEPTS = {
    "DCEA": r"\bdcea\b|distributional cost-effectiveness",
    "BIA": r"\bbia\b|budget impact",
    "VOI": r"\bvoi\b|value of information",
    "MCDA": r"\bmcda\b|multi-criteria",
    "Value of Harmonisation": r"value of harmonisation|harmonization|harmonisation",
    "policy brief": r"policy brief",
}

DOCUMENT_SUFFIXES = {".md", ".qmd", ".tex", ".rst", ".txt"}
SKIP_PARTS = {".git", ".conductor/local", "site", "dist", "build", "node_modules"}


@dataclass(frozen=True)
class ConceptHit:
    file: str
    line: int
    level: str
    concept: str
    text: str
    severity: str


def iter_documents(root: Path, explicit: list[Path] | None = None) -> Iterable[Path]:
    if explicit:
        for path in explicit:
            if path.exists() and path.is_file():
                yield path
        return
    roots = [root / "manuscripts", root / "manuscript", root / "docs", root / "README.md", root / "preprint.md", root / "paper.md"]
    seen: set[Path] = set()
    for candidate in roots:
        if candidate.is_file() and candidate.suffix.lower() in DOCUMENT_SUFFIXES:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                rel = path.relative_to(root).as_posix()
                if any(rel.startswith(part) for part in SKIP_PARTS):
                    continue
                if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES and path not in seen:
                    seen.add(path)
                    yield path


def heading_level(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return f"h{len(stripped) - len(stripped.lstrip('#'))}"
    if re.match(r"^\\(section|subsection|subsubsection)\{", stripped):
        if stripped.startswith("\\section"):
            return "h1"
        if stripped.startswith("\\subsection"):
            return "h2"
        return "h3"
    return "body"


def find_hits(path: Path, root: Path) -> list[ConceptHit]:
    hits: list[ConceptHit] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        level = heading_level(line)
        for concept, patterns in DEFAULT_ALLOWED_HEADLINE_CONCEPTS.items():
            if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in patterns):
                hits.append(ConceptHit(rel, line_no, level, concept, line.strip(), "ok"))
        for concept, pattern in SUPPORTING_CONCEPTS.items():
            if re.search(pattern, low, flags=re.IGNORECASE):
                severity = "warning" if level in {"h1", "h2"} else "info"
                hits.append(ConceptHit(rel, line_no, level, concept, line.strip(), severity))
    return hits


def audit(root: Path, docs: list[Path] | None = None) -> dict[str, object]:
    paths = list(iter_documents(root, docs))
    all_hits: list[ConceptHit] = []
    for path in paths:
        all_hits.extend(find_hits(path, root))
    headline_hits = [hit for hit in all_hits if hit.level in {"h1", "h2"}]
    allowed_headline = {hit.concept for hit in headline_hits if hit.concept in DEFAULT_ALLOWED_HEADLINE_CONCEPTS}
    supporting_headline = [hit for hit in headline_hits if hit.concept in SUPPORTING_CONCEPTS]
    warnings = [hit for hit in all_hits if hit.severity == "warning"]
    status = "pass"
    if len(allowed_headline) > 3 or supporting_headline:
        status = "review"
    supporting_counts: dict[str, int] = {}
    for hit in supporting_headline:
        key = hit.concept.lower()
        supporting_counts[key] = supporting_counts.get(key, 0) + 1
    report = {
        "status": status,
        "documents_scanned": [str(path.relative_to(root) if path.is_relative_to(root) else path) for path in paths],
        "allowed_headline_concepts_seen": sorted(allowed_headline),
        "supporting_concepts_in_headings": [asdict(hit) for hit in supporting_headline],
        "warning_count": len(warnings),
        "hits": [asdict(hit) for hit in all_hits],
        "budget": {
            "headline_concepts_allowed": sorted(DEFAULT_ALLOWED_HEADLINE_CONCEPTS),
            "supporting_concepts_should_not_be_headline": sorted(SUPPORTING_CONCEPTS),
        },
        "summary": {
            "status": status,
            "documents_scanned": len(paths),
            "issues_total": len(warnings),
            "supporting_heading_counts": supporting_counts,
        },
    }
    return report


def to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Concept scope budget audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Headline concept budget",
        "",
        "Allowed headline concepts:",
    ]
    for concept in report["budget"]["headline_concepts_allowed"]:  # type: ignore[index]
        lines.append(f"- {concept}")
    lines.extend(["", "Seen as headline concepts:"])
    seen = report.get("allowed_headline_concepts_seen", [])
    if seen:
        lines.extend(f"- {concept}" for concept in seen)  # type: ignore[assignment]
    else:
        lines.append("- None")
    lines.extend(["", "## Supporting concepts that appeared in H1/H2", ""])
    supporting = report.get("supporting_concepts_in_headings", [])
    if supporting:
        lines.append("| File | Line | Concept | Heading |")
        lines.append("|---|---:|---|---|")
        for hit in supporting:  # type: ignore[assignment]
            text = str(hit["text"]).replace("|", "\\|")
            lines.append(f"| `{hit['file']}` | {hit['line']} | {hit['concept']} | {text} |")
    else:
        lines.append("None detected.")
    lines.extend(["", "## All concept hits", ""])
    hits = report.get("hits", [])
    if hits:
        lines.append("| File | Line | Level | Concept | Severity | Text |")
        lines.append("|---|---:|---|---|---|---|")
        for hit in hits:  # type: ignore[assignment]
            text = str(hit["text"]).replace("|", "\\|")[:220]
            lines.append(f"| `{hit['file']}` | {hit['line']} | {hit['level']} | {hit['concept']} | {hit['severity']} | {text} |")
    else:
        lines.append("No concept terms detected in scanned documents.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--document", action="append", type=Path, default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    docs = [path.resolve() for path in args.document] if args.document else None
    report = audit(root, docs)
    local_dir = root / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (local_dir / "concept_budget.json")
    output_md = args.output_md or (local_dir / "concept_budget.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(to_markdown(report), encoding="utf-8")
    print(output_md)
    if args.strict and report["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Bootstrap compatibility alias.
build_budget = audit
