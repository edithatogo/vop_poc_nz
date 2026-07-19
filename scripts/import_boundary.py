#!/usr/bin/env python3
"""Check cross-repository import boundaries for vop_poc_nz and voiage.

The intended architecture is contract-first: vop_poc_nz can produce manifests,
Arrow/Parquet/CSV/JSON fixtures and optional adapter modules, while voiage owns
production Value of Perspective methods. This tool prevents accidental hard
coupling between the repositories.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SKIP_DIRS = {".git", ".venv", ".pixi", "node_modules", "dist", "build", ".conductor", "site", "_site"}
DEFAULT_ALLOWED_VOP_IMPORT_PATHS = (
    "src/vop_poc_nz/adapters/",
    "src/vop_poc_nz/integrations/",
    "tests/",
    "examples/",
)


@dataclass(frozen=True)
class ImportIssue:
    path: str
    line: int
    severity: str
    imported: str
    message: str


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def detect_project(root: Path) -> str:
    if (root / "src" / "vop_poc_nz").exists() or root.name == "vop_poc_nz":
        return "vop_poc_nz"
    if (root / "src" / "voiage").exists() or root.name == "voiage":
        return "voiage"
    return "unknown"


def iter_py_files(root: Path) -> Iterable[Path]:
    for current_root, dirs, files in root.walk():
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            path = Path(current_root) / name
            if path.suffix == ".py":
                yield path


def imported_names(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.append((node.lineno, node.module))
    return found


def path_allowed(rel: str, allowed_prefixes: tuple[str, ...]) -> bool:
    normalised = rel.replace("\\", "/")
    return any(normalised.startswith(prefix) for prefix in allowed_prefixes)


def build_report(root: Path, allowed_vop_import_paths: tuple[str, ...] = DEFAULT_ALLOWED_VOP_IMPORT_PATHS) -> dict[str, object]:
    root = root.resolve()
    kind = detect_project(root)
    issues: list[ImportIssue] = []
    imports_seen: dict[str, int] = {}
    for path in iter_py_files(root):
        rel = rel_posix(path, root)
        for line, name in imported_names(path):
            top = name.split(".", 1)[0]
            imports_seen[top] = imports_seen.get(top, 0) + 1
            if kind == "vop_poc_nz" and top == "voiage" and not path_allowed(rel, allowed_vop_import_paths):
                issues.append(
                    ImportIssue(
                        rel,
                        line,
                        "error",
                        name,
                        "vop_poc_nz should not hard-import voiage except in explicit adapters/tests/examples",
                    )
                )
            elif kind == "voiage" and top == "vop_poc_nz":
                issues.append(
                    ImportIssue(
                        rel,
                        line,
                        "error",
                        name,
                        "voiage must remain general and must not import the NZ proof-of-concept package",
                    )
                )
    return {
        "schema_version": "1.0",
        "repo": {"root": str(root), "name": root.name, "detected_project": kind},
        "summary": {
            "python_files_scanned": sum(1 for _ in iter_py_files(root)),
            "issues_total": len(issues),
            "boundary_pass": len(issues) == 0,
            "top_level_import_counts": dict(sorted(imports_seen.items())),
        },
        "issues": [asdict(issue) for issue in issues],
        "recommendations": [
            "Use file/schema/CLI boundaries rather than direct cross-package imports for production integration.",
            "Keep optional adapters isolated in src/vop_poc_nz/adapters or examples until the public API is stable.",
        ],
    }


def to_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [f"# Import boundary report: {report['repo']['name']}", "", "## Summary", ""]
    for key in ("python_files_scanned", "issues_total", "boundary_pass"):
        lines.append(f"- `{key}`: {summary[key]}")
    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    if report["issues"]:
        lines.extend(["", "## Issues", "", "| Severity | File | Line | Import | Message |", "|---|---|---:|---|---|"])
        for issue in report["issues"]:
            lines.append(f"| `{issue['severity']}` | `{issue['path']}` | {issue['line']} | `{issue['imported']}` | {issue['message']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = build_report(args.repo)
    out_dir = args.repo / ".conductor" / "local"
    output_json = args.output_json or out_dir / "import_boundary.json"
    output_md = args.output_md or out_dir / "import_boundary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Import boundary report written: {output_md}")
    if args.strict and not report["summary"]["boundary_pass"]:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
