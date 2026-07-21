"""Prepare a deterministic, fail-closed arXiv source bundle.

Inspired by arXivIt, arxiv-latex-cleaner and ALC-NG, but intentionally
stdlib-only and non-mutating: source files remain untouched and all generated
material is written to an explicit output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

FORBIDDEN = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".synctex.gz",
}
REFERENCE_SUFFIXES = (".tex", ".bib", ".pdf", ".png", ".jpg", ".jpeg", ".svg")
PATTERNS = [
    re.compile(r"\\(?:input|include)\{([^}]+)\}"),
    re.compile(r"\\(?:addbibresource|bibliography)\{([^}]+)\}"),
    re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}"),
]


def resolve_reference(source: Path, raw: str) -> Path:
    """Resolve a local LaTeX reference without searching outside source."""
    candidate = (source / raw).resolve()
    if source not in candidate.parents:
        raise SystemExit(f"reference escapes source directory: {raw}")
    if not candidate.suffix:
        matches = [source / f"{raw}{suffix}" for suffix in REFERENCE_SUFFIXES]
        candidate = next(
            (path.resolve() for path in matches if path.is_file()), candidate
        )
    if not candidate.is_file():
        raise SystemExit(f"unresolved reference: {raw}")
    return candidate


def discover_references(source: Path, main_file: Path) -> set[Path]:
    """Return the main file and every directly referenced local asset."""
    references = {main_file}
    text = main_file.read_text(encoding="utf-8")
    for pattern in PATTERNS:
        references.update(
            resolve_reference(source, raw) for raw in pattern.findall(text)
        )
    return references


def validate_source_inputs(source: Path) -> None:
    """Reject hidden and generated files outside known build directories."""
    for path in source.rglob("*"):
        if any(part.startswith("build") for part in path.relative_to(source).parts):
            continue
        if path.is_file() and (
            path.name.startswith(".") or path.suffix.lower() in FORBIDDEN
        ):
            raise SystemExit(f"forbidden submission input: {path.relative_to(source)}")


def copy_references(
    source: Path, output: Path, references: set[Path]
) -> list[dict[str, str]]:
    """Copy resolved references and return their content-addressed records."""
    records: list[dict[str, str]] = []
    for path in sorted(references):
        relative = path.relative_to(source)
        if " " in relative.as_posix():
            raise SystemExit(f"unsafe filename (space): {relative}")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("manuscript"))
    ap.add_argument("--main", default="jss_submission.tex")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    src = args.source.resolve()
    out = args.output.resolve()
    main_file = src / args.main
    if not main_file.is_file():
        raise SystemExit(f"missing main source: {main_file}")
    if out == src or src in out.parents:
        raise SystemExit("output must be outside manuscript source")
    refs = discover_references(src, main_file)
    validate_source_inputs(src)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    records = copy_references(src, out, refs)
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=src.parent, text=True
        ).strip()
    except OSError, subprocess.CalledProcessError:
        rev = "unavailable"
    manifest = {
        "format": 1,
        "main": args.main,
        "source_revision": rev,
        "files": records,
        "policy": "arxivIt+arxiv-latex-cleaner+ALC-NG-inspired",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
