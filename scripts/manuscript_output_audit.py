#!/usr/bin/env python3
"""Audit manuscript outputs against result manifests and claim-boundary phrases."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

OUTPUT_REF_RE = re.compile(r"\b(Figure|Table)\s+([0-9]+[A-Za-z]?)\b")
QUANT_CLAIM_RE = re.compile(
    r"\b(EVoP|VoP|ICER|NMB|QALY|dominant|cost-saving|cost effective|cost-effective|empirical demonstration|societal bonus)\b",
    re.IGNORECASE,
)
DOCUMENT_SUFFIXES = {".md", ".qmd", ".tex", ".rst", ".txt"}


@dataclass(frozen=True)
class OutputReference:
    file: str
    line: int
    ref_type: str
    ref_id: str
    text: str
    manifest_status: str


@dataclass(frozen=True)
class ClaimReference:
    file: str
    line: int
    phrase: str
    text: str
    severity: str


def iter_documents(root: Path, explicit: list[Path] | None = None) -> Iterable[Path]:
    if explicit:
        yield from explicit
        return
    for base in [root / "manuscripts", root / "docs", root / "README.md", root / "paper.md", root / "preprint.md"]:
        if base.is_file() and base.suffix.lower() in DOCUMENT_SUFFIXES:
            yield base
        elif base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES and ".conductor/local" not in path.as_posix():
                    yield path


def load_manifest_refs(manifest_paths: list[Path]) -> set[str]:
    refs: set[str] = set()
    for path in manifest_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payloads = data if isinstance(data, list) else [data]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            for key in ("output_id", "artifact_id", "table_id", "figure_id", "claim_id"):
                value = item.get(key)
                if value:
                    refs.add(str(value).lower())
            if isinstance(item.get("outputs"), list):
                for output in item["outputs"]:
                    if isinstance(output, dict):
                        for key in ("output_id", "artifact_id", "table_id", "figure_id", "claim_id"):
                            value = output.get(key)
                            if value:
                                refs.add(str(value).lower())
    return refs


def discover_manifests(root: Path) -> list[Path]:
    return sorted(
        {
            *root.glob("results/**/*manifest*.json"),
            *root.glob("outputs/**/*manifest*.json"),
            *root.glob("artifacts/**/*manifest*.json"),
            *root.glob(".conductor/local/*manifest*.json"),
        }
    )


def audit(root: Path, documents: list[Path] | None = None, manifests: list[Path] | None = None) -> dict[str, Any]:
    root = root.resolve()
    docs = [path.resolve() for path in documents] if documents else list(iter_documents(root))
    manifest_paths = [path.resolve() for path in manifests] if manifests else discover_manifests(root)
    manifest_refs = load_manifest_refs(manifest_paths)
    outputs: list[OutputReference] = []
    claims: list[ClaimReference] = []
    for path in docs:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in OUTPUT_REF_RE.finditer(line):
                ref_type, number = match.groups()
                candidates = {
                    f"{ref_type.lower()}_{number.lower()}",
                    f"{ref_type.lower()}{number.lower()}",
                    f"{ref_type.lower()} {number.lower()}",
                }
                status = "manifest_backed" if candidates & manifest_refs else "needs_manifest"
                outputs.append(OutputReference(rel, line_no, ref_type, number, line.strip(), status))
            for match in QUANT_CLAIM_RE.finditer(line):
                phrase = match.group(1)
                lower = phrase.lower()
                severity = "warning" if lower in {"empirical demonstration", "societal bonus", "dominant", "cost-saving"} else "info"
                claims.append(ClaimReference(rel, line_no, phrase, line.strip(), severity))
    return {
        "documents_scanned": [path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path) for path in docs],
        "manifests_scanned": [path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path) for path in manifest_paths],
        "outputs": [asdict(item) for item in outputs],
        "claims": [asdict(item) for item in claims],
        "needs_manifest_count": sum(1 for item in outputs if item.manifest_status == "needs_manifest"),
        "warning_claim_count": sum(1 for item in claims if item.severity == "warning"),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Manuscript output audit",
        "",
        f"- Documents scanned: {len(report.get('documents_scanned', []))}",
        f"- Manifests scanned: {len(report.get('manifests_scanned', []))}",
        f"- Output references needing manifests: {report.get('needs_manifest_count', 0)}",
        f"- Warning claim phrases: {report.get('warning_claim_count', 0)}",
        "",
        "## Output references",
        "",
    ]
    outputs = report.get("outputs", [])
    if outputs:
        lines.append("| File | Line | Ref | Status | Text |")
        lines.append("|---|---:|---|---|---|")
        for item in outputs:
            text = str(item["text"]).replace("|", "\\|")[:220]
            lines.append(f"| `{item['file']}` | {item['line']} | {item['ref_type']} {item['ref_id']} | {item['manifest_status']} | {text} |")
    else:
        lines.append("No figure/table references detected.")
    lines.extend(["", "## Quantitative or claim-boundary phrases", ""])
    claims = report.get("claims", [])
    if claims:
        lines.append("| File | Line | Phrase | Severity | Text |")
        lines.append("|---|---:|---|---|---|")
        for item in claims:
            text = str(item["text"]).replace("|", "\\|")[:220]
            lines.append(f"| `{item['file']}` | {item['line']} | `{item['phrase']}` | {item['severity']} | {text} |")
    else:
        lines.append("No target claim phrases detected.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--document", action="append", type=Path, default=[])
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    report = audit(root, documents=args.document or None, manifests=args.manifest or None)
    local_dir = root / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "manuscript_output_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (local_dir / "manuscript_output_audit.md").write_text(to_markdown(report), encoding="utf-8")
    print(local_dir / "manuscript_output_audit.md")
    if args.strict and report.get("needs_manifest_count", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
