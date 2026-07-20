#!/usr/bin/env python3
"""Reconcile manuscript outputs against local result manifests.

This tool is intentionally conservative. It does not assume that every figure or
claim in a manuscript can be resolved automatically. Instead, it builds a
machine-readable checklist for the local agent:

* which manuscripts were scanned;
* which tables/figures/equations were referenced;
* which result manifests were found;
* which references have explicit provenance anchors; and
* which quantitative or normative claims need review before a preprint/journal
  update.

Supported explicit anchors in Markdown/Quarto/LaTeX comments:

    <!-- result: figure_5; artifact=results/figure_5.png; manifest=results/result_manifest.json -->
    % result: table_2; artifact=results/table_2.csv

The output belongs in .conductor/local and should generally not be committed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = {".md", ".qmd", ".tex", ".rst", ".txt"}
MANUSCRIPT_DIRS = ("manuscript", "manuscripts", "paper", "papers", "preprint", "docs")
SKIP_DIRS = {".git", ".venv", ".pixi", "node_modules", "site", "_site", "dist", "build", ".conductor/local"}

REFERENCE_RE = re.compile(r"\b(?P<kind>Figure|Fig\.?|Table|Eq\.?|Equation)\s*~?\??(?P<label>[A-Za-z0-9_.:-]+)", re.IGNORECASE)
ANCHOR_RE = re.compile(r"(?:<!--|%)\s*result:\s*(?P<id>[A-Za-z0-9_.:-]+)\s*(?:;(?P<body>.*?))?(?:-->)?\s*$", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:NZ\$|\$)?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|/QALY|QALYs?|NZD|years?|iterations?|N\s*=)?", re.IGNORECASE)
CLAIM_TERMS = {
    "societal bonus": "replace or define as a decomposed perspective increment, not a headline method",
    "empirical demonstration": "use only when all inputs and outputs are backed by evidence ledgers/model cards",
    "hypothetical": "avoid if empirically parameterised; use case grade labels instead",
    "validated model": "requires explicit validation evidence",
    "policy-grade": "requires model card, evidence ledger, and validation status",
    "dominant": "define at first use and verify signs for incremental costs/effects",
    "cost-saving": "requires source artifact and sign convention review",
    "first open-source": "requires literature/software search evidence",
    "stakeholder preferences": "requires elicitation/source method or label as scenario weights",
    "value of harmonisation": "keep scoped as extension unless it becomes a main concept",
}


@dataclass(frozen=True)
class TextFile:
    path: str
    lines: int
    words: int


@dataclass(frozen=True)
class Reference:
    path: str
    line: int
    kind: str
    label: str
    text: str


@dataclass(frozen=True)
class Anchor:
    path: str
    line: int
    result_id: str
    artifact: str | None
    manifest: str | None
    raw: str


@dataclass(frozen=True)
class ClaimHit:
    path: str
    line: int
    term: str
    guidance: str
    text: str


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_candidate_files(root: Path, include_all_text: bool = False) -> Iterable[Path]:
    for current_root, dirs, files in Path(root).resolve().walk():
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        current = Path(current_root)
        if not include_all_text:
            parts = set(current.relative_to(root).parts) if current != root else set()
            if current != root and not (parts & set(MANUSCRIPT_DIRS)):
                # Still allow root-level manuscript files.
                pass
        for name in files:
            path = current / name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel = rel_posix(path, root)
            if include_all_text or rel.split("/", 1)[0] in MANUSCRIPT_DIRS or path.parent == root:
                yield path


def parse_anchor_body(body: str | None) -> tuple[str | None, str | None]:
    artifact = None
    manifest = None
    if not body:
        return artifact, manifest
    for part in body.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip().strip('"\'')
        if key == "artifact":
            artifact = value
        elif key == "manifest":
            manifest = value
    return artifact, manifest


def scan_text_file(path: Path, root: Path) -> tuple[TextFile, list[Reference], list[Anchor], list[ClaimHit], list[dict[str, object]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = rel_posix(path, root)
    lines = text.splitlines()
    refs: list[Reference] = []
    anchors: list[Anchor] = []
    claims: list[ClaimHit] = []
    numeric_claims: list[dict[str, object]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        for match in REFERENCE_RE.finditer(line):
            refs.append(
                Reference(
                    path=rel,
                    line=idx,
                    kind=match.group("kind"),
                    label=match.group("label"),
                    text=stripped[:240],
                )
            )
        anchor_match = ANCHOR_RE.search(line)
        if anchor_match:
            artifact, manifest = parse_anchor_body(anchor_match.group("body"))
            anchors.append(
                Anchor(
                    path=rel,
                    line=idx,
                    result_id=anchor_match.group("id"),
                    artifact=artifact,
                    manifest=manifest,
                    raw=stripped[:240],
                )
            )
        lower = line.lower()
        for term, guidance in CLAIM_TERMS.items():
            if term in lower:
                claims.append(ClaimHit(path=rel, line=idx, term=term, guidance=guidance, text=stripped[:240]))
        if any(token in lower for token in ("icer", "evop", "vop", "qaly", "nmb", "wtp", "threshold", "discount", "iterations")):
            numbers = [m.group(0).strip() for m in NUMBER_RE.finditer(line)]
            if numbers:
                numeric_claims.append({"path": rel, "line": idx, "numbers": numbers[:20], "text": stripped[:240]})
    words = len(re.findall(r"\w+", text))
    return TextFile(path=rel, lines=len(lines), words=words), refs, anchors, claims, numeric_claims


def find_result_manifests(root: Path) -> list[dict[str, object]]:
    patterns = [
        "**/result_manifest*.json",
        "**/*result*manifest*.json",
        "**/release_snapshot.json",
    ]
    seen: set[Path] = set()
    records: list[dict[str, object]] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
                continue
            if path in seen:
                continue
            seen.add(path)
            item: dict[str, object] = {"path": rel_posix(path, root), "parse_status": "ok", "outputs": []}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - exact errors vary
                item["parse_status"] = f"error: {exc}"
                records.append(item)
                continue
            outputs: list[str] = []
            for key in ("outputs", "artifacts", "files"):
                value = data.get(key) if isinstance(data, dict) else None
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, str):
                            outputs.append(entry)
                        elif isinstance(entry, dict):
                            for k in ("path", "output", "artifact", "file"):
                                if isinstance(entry.get(k), str):
                                    outputs.append(entry[k])
                                    break
            item["outputs"] = sorted(set(outputs))
            records.append(item)
    return records


def build_reconciliation(root: Path, include_all_text: bool = False) -> dict[str, object]:
    root = root.resolve()
    files: list[TextFile] = []
    references: list[Reference] = []
    anchors: list[Anchor] = []
    claims: list[ClaimHit] = []
    numeric_claims: list[dict[str, object]] = []
    for path in sorted(iter_candidate_files(root, include_all_text=include_all_text)):
        file_record, refs, found_anchors, found_claims, found_numeric = scan_text_file(path, root)
        files.append(file_record)
        references.extend(refs)
        anchors.extend(found_anchors)
        claims.extend(found_claims)
        numeric_claims.extend(found_numeric)

    manifests = find_result_manifests(root)
    anchor_ids = {anchor.result_id.lower() for anchor in anchors}
    unresolved = []
    for ref in references:
        normalised = f"{ref.kind.lower().replace('.', '')}_{ref.label}".lower().replace(":", "_")
        label_only = ref.label.lower().replace(":", "_")
        if normalised not in anchor_ids and label_only not in anchor_ids:
            unresolved.append(asdict(ref))

    term_counts = Counter(hit.term for hit in claims)
    recommendations: list[str] = []
    if unresolved:
        recommendations.append("Add explicit result anchors for manuscript tables/figures or map them in a result manifest.")
    if numeric_claims:
        recommendations.append("Ensure every numeric claim in the manuscript has a result manifest or evidence-ledger source.")
    if term_counts:
        recommendations.append("Run claim-boundary review for flagged terms before arXiv/journal submission.")
    if not manifests:
        recommendations.append("No result manifests were found; run the no-orphan-result pipeline before release.")

    return {
        "schema_version": "1.0",
        "repo": {"root": str(root), "name": root.name},
        "summary": {
            "text_files_scanned": len(files),
            "references_found": len(references),
            "result_anchors_found": len(anchors),
            "result_manifests_found": len(manifests),
            "unresolved_references": len(unresolved),
            "claim_terms_found": dict(term_counts),
            "numeric_claim_lines": len(numeric_claims),
        },
        "text_files": [asdict(item) for item in files],
        "references": [asdict(item) for item in references],
        "result_anchors": [asdict(item) for item in anchors],
        "result_manifests": manifests,
        "unresolved_references": unresolved,
        "claim_hits": [asdict(item) for item in claims],
        "numeric_claims": numeric_claims,
        "recommendations": recommendations,
    }


def to_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        f"# Manuscript/output reconciliation: {report['repo']['name']}",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    if report["unresolved_references"]:
        lines.extend(["", "## Unresolved table/figure/equation references", "", "| File | Line | Reference | Text |", "|---|---:|---|---|"])
        for item in report["unresolved_references"][:200]:
            lines.append(f"| `{item['path']}` | {item['line']} | {item['kind']} {item['label']} | {item['text']} |")
    if report["claim_hits"]:
        lines.extend(["", "## Claim-boundary hits", "", "| File | Line | Term | Guidance |", "|---|---:|---|---|"])
        for item in report["claim_hits"][:200]:
            lines.append(f"| `{item['path']}` | {item['line']} | `{item['term']}` | {item['guidance']} |")
    if report["numeric_claims"]:
        lines.extend(["", "## Numeric claim lines", "", "| File | Line | Numbers | Text |", "|---|---:|---|---|"])
        for item in report["numeric_claims"][:200]:
            lines.append(f"| `{item['path']}` | {item['line']} | `{', '.join(item['numbers'])}` | {item['text']} |")
    if report["result_anchors"]:
        lines.extend(["", "## Explicit result anchors", "", "| File | Line | Result ID | Artifact | Manifest |", "|---|---:|---|---|---|"])
        for item in report["result_anchors"]:
            lines.append(f"| `{item['path']}` | {item['line']} | `{item['result_id']}` | `{item['artifact']}` | `{item['manifest']}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--include-all-text", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Return non-zero if unresolved references or claim hits are found")
    args = parser.parse_args()

    report = build_reconciliation(args.repo, include_all_text=args.include_all_text)
    default_dir = args.repo / ".conductor" / "local"
    output_json = args.output_json or default_dir / "manuscript_reconciliation.json"
    output_md = args.output_md or default_dir / "manuscript_reconciliation.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Manuscript reconciliation written: {output_md}")
    if args.strict and (report["summary"]["unresolved_references"] or report["summary"]["claim_terms_found"]):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
