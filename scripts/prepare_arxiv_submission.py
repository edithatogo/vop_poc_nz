"""Prepare a deterministic, fail-closed arXiv source bundle.

Inspired by arXivIt, arxiv-latex-cleaner and ALC-NG, but intentionally
stdlib-only and non-mutating: source files remain untouched and all generated
material is written to an explicit output directory.
"""
from __future__ import annotations

import argparse, hashlib, json, re, shutil, subprocess
from pathlib import Path

FORBIDDEN = {".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf", ".synctex.gz"}
PATTERNS = [re.compile(r"\\(?:input|include)\{([^}]+)\}"), re.compile(r"\\addbibresource\{([^}]+)\}"), re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("manuscript"))
    ap.add_argument("--main", default="jss_submission.tex")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    main_file = src / args.main
    if not main_file.is_file():
        raise SystemExit(f"missing main source: {main_file}")
    if out == src or src in out.parents:
        raise SystemExit("output must be outside manuscript source")
    refs = {main_file}
    text = main_file.read_text(encoding="utf-8")
    for pattern in PATTERNS:
        for raw in pattern.findall(text):
            candidate = (src / raw)
            if not candidate.suffix:
                for suffix in (".tex", ".bib", ".pdf", ".png", ".jpg", ".jpeg", ".svg"):
                    if (src / (raw + suffix)).exists(): candidate = src / (raw + suffix); break
            if not candidate.is_file(): raise SystemExit(f"unresolved reference: {raw}")
            refs.add(candidate.resolve())
    for p in src.rglob("*"):
        if "build" in p.relative_to(src).parts:
            continue
        if p.is_file() and (p.name.startswith(".") or p.suffix.lower() in FORBIDDEN):
            raise SystemExit(f"forbidden submission input: {p.relative_to(src)}")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    records = []
    for p in sorted(refs):
        rel = p.relative_to(src)
        if " " in rel.as_posix(): raise SystemExit(f"unsafe filename (space): {rel}")
        target = out / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
        records.append({"path": rel.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    try: rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src.parent, text=True).strip()
    except Exception: rev = "unavailable"
    manifest = {"format": 1, "main": args.main, "source_revision": rev, "files": records, "policy": "arxivIt+arxiv-latex-cleaner+ALC-NG-inspired"}
    args.manifest.parent.mkdir(parents=True, exist_ok=True); args.manifest.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
