#!/usr/bin/env python3
"""Map a local repository before applying conductor overlays.

The mapper is deliberately dependency-light. It can run in a newly cloned repo
before Pixi/conda/pip dependencies are installed. It produces a JSON inventory
and a human-readable Markdown summary that downstream coding agents can use to
triage what should be committed, kept local, regenerated, or reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "htmlcov",
    ".pixi",
}

PUBLIC_ROOT_FILES = {
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "pyproject.toml",
    "pixi.toml",
    "pixi.lock",
    "tox.ini",
    "mkdocs.yml",
    "ruff.toml",
    "uv.lock",
}

SOURCE_PREFIXES = (
    "src/",
    "tests/",
    "docs/",
    "conductor/",
    "schemas/",
    "examples/",
    "templates/",
    "scripts/",
    "adr/",
    "issues/",
    "prompts/",
    ".github/",
)

LOCAL_ONLY_PREFIXES = (
    ".conductor/local/",
    ".local/",
    "local/",
    "private/",
    "secrets/",
    "data/raw/",
    "data/private/",
    "manuscripts/submissions/",
    "reviewer_letters/",
    "artifacts/local/",
    "outputs/local/",
    "results/local/",
)

REVIEW_PREFIXES = (
    "data/",
    "notebooks/",
    "manuscripts/",
    "reports/",
    "results/",
    "outputs/",
    "artifacts/",
    "figures/",
    "site/",
)

GENERATED_PATTERNS = (
    ".egg-info/",
    ".ipynb_checkpoints/",
    ".coverage",
    "coverage.xml",
    "pytestdebug.log",
)

SECRET_TOKENS = (
    ".env",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "token",
    "apikey",
    "api_key",
    "private_key",
)

LOCAL_ONLY_SUFFIXES = (
    ".pkl",
    ".pickle",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".feather",
    ".arrow",
    ".parquet",
)

NEEDS_REVIEW_SUFFIXES = (
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".csv",
    ".tsv",
    ".ipynb",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".svg",
)


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str
    tracked: bool
    category: str
    publish_policy: str
    reasons: list[str]


def run_git(root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_repo_files(root: Path) -> Iterable[Path]:
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for name in files:
            path = current / name
            if path.is_file():
                yield path


def tracked_files(root: Path) -> set[str]:
    output = run_git(root, ["ls-files"])
    if output is None or output == "":
        return set()
    return set(output.splitlines())


def detect_project(root: Path) -> dict[str, object]:
    package_dirs: list[str] = []
    for candidate in (root / "src").glob("*") if (root / "src").exists() else []:
        if candidate.is_dir():
            package_dirs.append(candidate.name)
    name = root.name
    if (root / "src" / "vop_poc_nz").exists() or name == "vop_poc_nz":
        kind = "vop_poc_nz"
    elif (root / "src" / "voiage").exists() or name == "voiage":
        kind = "voiage"
    else:
        kind = "unknown"
    return {"kind": kind, "package_dirs": package_dirs}


def classify(path_string: str) -> tuple[str, str, list[str]]:
    p = path_string.replace("\\", "/")
    lower = p.lower()
    reasons: list[str] = []

    if any(token in lower for token in SECRET_TOKENS):
        reasons.append("name suggests secret/credential/local environment")
        return "local_only", "do_not_commit", reasons

    if any(lower.startswith(prefix) for prefix in LOCAL_ONLY_PREFIXES):
        reasons.append("path is in a local/private/raw workspace")
        return "local_only", "do_not_commit", reasons

    if any(pattern in lower for pattern in GENERATED_PATTERNS):
        reasons.append("path looks generated/cache-like")
        return "generated", "do_not_commit", reasons

    suffix = Path(p).suffix.lower()
    if suffix in LOCAL_ONLY_SUFFIXES:
        reasons.append("binary/intermediate data output should normally be regenerated")
        return "local_only", "do_not_commit", reasons

    if Path(p).name in PUBLIC_ROOT_FILES:
        reasons.append("standard public repository metadata/configuration")
        return "source_public", "commit", reasons

    if lower.startswith(SOURCE_PREFIXES):
        if suffix in NEEDS_REVIEW_SUFFIXES and not lower.startswith(("docs/", "examples/")):
            reasons.append("file is in a public area but binary/data/notebook content needs review")
            return "needs_review", "review_before_commit", reasons
        reasons.append("source/test/docs/conductor-style path")
        return "source_public", "commit", reasons

    if lower.startswith(REVIEW_PREFIXES) or suffix in NEEDS_REVIEW_SUFFIXES:
        reasons.append("data/output/manuscript/binary content requires publication triage")
        return "needs_review", "review_before_commit", reasons

    if lower.endswith((".tmp", ".bak", ".log")):
        reasons.append("temporary or log file")
        return "generated", "do_not_commit", reasons

    reasons.append("unclassified; agent should inspect before publishing")
    return "needs_review", "review_before_commit", reasons


def build_map(root: Path) -> dict[str, object]:
    root = root.resolve()
    tracked = tracked_files(root)
    records: list[FileRecord] = []
    for file_path in sorted(iter_repo_files(root), key=lambda p: rel_posix(p, root)):
        rel = rel_posix(file_path, root)
        category, policy, reasons = classify(rel)
        records.append(
            FileRecord(
                path=rel,
                size_bytes=file_path.stat().st_size,
                sha256=sha256_file(file_path),
                tracked=rel in tracked,
                category=category,
                publish_policy=policy,
                reasons=reasons,
            )
        )

    category_counts = Counter(record.category for record in records)
    policy_counts = Counter(record.publish_policy for record in records)
    branch = run_git(root, ["branch", "--show-current"])
    head = run_git(root, ["rev-parse", "--short", "HEAD"])
    status = run_git(root, ["status", "--short"])
    tracked_bad = [
        r.path
        for r in records
        if r.tracked and r.publish_policy in {"do_not_commit", "review_before_commit"}
    ]

    recommendations: list[str] = []
    if tracked_bad:
        recommendations.append(
            "Review tracked files that are generated, local-only, or publication-sensitive before pushing."
        )
    if category_counts.get("needs_review", 0):
        recommendations.append(
            "Create or update a publication-policy manifest for data, manuscripts, outputs, and notebooks."
        )
    if category_counts.get("local_only", 0):
        recommendations.append(
            "Move local-only work into ignored local workspaces or remove it from the git index."
        )

    return {
        "schema_version": "1.0",
        "repo": {
            "root": str(root),
            "name": root.name,
            "branch": branch,
            "head": head,
            "is_git_repo": head is not None,
            "dirty": bool(status),
        },
        "detected_project": detect_project(root),
        "summary": {
            "total_files": len(records),
            "tracked_files": sum(1 for r in records if r.tracked),
            "category_counts": dict(category_counts),
            "publish_policy_counts": dict(policy_counts),
            "tracked_files_needing_review_or_exclusion": tracked_bad,
        },
        "recommendations": recommendations,
        "files": [asdict(record) for record in records],
    }


def to_markdown(mapping: dict[str, object]) -> str:
    repo = mapping["repo"]
    detected = mapping["detected_project"]
    summary = mapping["summary"]
    lines = [
        f"# Repository map: {repo['name']}",
        "",
        f"- Root: `{repo['root']}`",
        f"- Git repo: `{repo['is_git_repo']}`",
        f"- Branch: `{repo['branch']}`",
        f"- HEAD: `{repo['head']}`",
        f"- Dirty: `{repo['dirty']}`",
        f"- Detected project: `{detected['kind']}`",
        "",
        "## Summary",
        "",
        f"- Total files scanned: {summary['total_files']}",
        f"- Tracked files: {summary['tracked_files']}",
        "",
        "### Categories",
        "",
    ]
    for key, value in sorted(summary["category_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "### Publish policies", ""])
    for key, value in sorted(summary["publish_policy_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    tracked_bad = summary["tracked_files_needing_review_or_exclusion"]
    if tracked_bad:
        lines.extend(["", "## Tracked files requiring review/exclusion", ""])
        for path in tracked_bad[:200]:
            lines.append(f"- `{path}`")
        if len(tracked_bad) > 200:
            lines.append(f"- ... plus {len(tracked_bad) - 200} more")
    lines.extend(["", "## Recommendations", ""])
    for item in mapping.get("recommendations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## File inventory", ""])
    lines.append("| Path | Category | Policy | Tracked | Reasons |")
    lines.append("|---|---|---:|---:|---|")
    for record in mapping["files"][:500]:
        reason = "; ".join(record["reasons"])
        lines.append(
            f"| `{record['path']}` | `{record['category']}` | `{record['publish_policy']}` | {record['tracked']} | {reason} |"
        )
    if len(mapping["files"]) > 500:
        lines.append(f"| ... | ... | ... | ... | {len(mapping['files']) - 500} more files omitted |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="Repository root to map")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    mapping = build_map(args.repo)
    json_text = json.dumps(mapping, indent=2, sort_keys=True)
    md_text = to_markdown(mapping)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text + "\n", encoding="utf-8")
    else:
        print(json_text)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
