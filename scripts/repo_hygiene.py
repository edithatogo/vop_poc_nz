#!/usr/bin/env python3
"""Generate a non-destructive repository-root hygiene and migration plan."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HygieneItem:
    path: str
    category: str
    action: str
    reason: str
    tracked: bool


def tracked_files(repo: Path) -> set[str]:
    try:
        output = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout
        return {item.decode("utf-8") for item in output.split(b"\0") if item}
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return set()


def classify(path: Path, repo: Path, tracked: set[str]) -> HygieneItem | None:
    rel = str(path.relative_to(repo))
    name = path.name.casefold()
    if path.parent != repo:
        return None
    is_tracked = rel in tracked
    if re.match(r"^(debug|full_log|run_\d+_log|.*_log_latest).*\.(txt|log)$", name):
        return HygieneItem(rel, "debug_log", "move_local_and_untrack", "debug/run logs do not belong at repository root", is_tracked)
    if name.endswith((".zip", ".tar", ".tar.gz", ".whl")):
        return HygieneItem(rel, "archive", "externalise_or_release_asset", "source/output archives should be release assets or local artifacts", is_tracked)
    if name.endswith(("_backup.md", "_backup.txt")) or name in {"plan_backup.md", "todo_backup.md"}:
        return HygieneItem(rel, "backup", "remove_after_history_check", "Git history replaces ad-hoc backup files", is_tracked)
    if name in {"site", "htmlcov", "output", "outputs", "results", ".snakemake"} and path.is_dir():
        return HygieneItem(rel, "generated_directory", "ignore_or_publish_externally", "generated build/output directory", is_tracked)
    if name in {"manuscript", "pptx", "references"} and path.is_dir():
        return HygieneItem(rel, "publication_asset", "review_publication_boundary", "may be valid in a research compendium but requires explicit public/local decision", is_tracked)
    if name.startswith("check_") and name.endswith(".py") or name.startswith("reproduce_") and name.endswith(".py"):
        return HygieneItem(rel, "one_off_script", "move_to_scripts_or_tests", "one-off scripts should have a named scripts/tests home", is_tracked)
    return None


def build_plan(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    tracked = tracked_files(repo)
    items = [item for path in repo.iterdir() if (item := classify(path, repo, tracked)) is not None]
    return {
        "schema_version": "1.0",
        "repo": str(repo),
        "summary": {
            "items": len(items),
            "tracked_items": sum(item.tracked for item in items),
            "root_cleanup_needed": bool(items),
        },
        "items": [asdict(item) for item in items],
        "rule": "This tool produces a plan only. Review artifact promotion, git history, and publication needs before moving or deleting anything.",
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Repository hygiene plan", "", f"Items requiring review: **{report['summary']['items']}** · Tracked: **{report['summary']['tracked_items']}**", "", "| Path | Category | Proposed action | Tracked | Reason |", "|---|---|---|---|---|"]
    for item in report["items"]:
        lines.append(f"| `{item['path']}` | {item['category']} | {item['action']} | {item['tracked']} | {item['reason']} |")
    if not report["items"]:
        lines.append("| — | — | — | — | No root-level hygiene candidates detected |")
    lines += ["", f"> {report['rule']}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    report = build_plan(args.repo)
    out = args.repo / ".conductor" / "local"
    out.mkdir(parents=True, exist_ok=True)
    (out / "repo_hygiene.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "repo_hygiene.md").write_text(to_markdown(report), encoding="utf-8")
    print(out / "repo_hygiene.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
