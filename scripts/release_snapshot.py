#!/usr/bin/env python3
"""Build a local release/readiness snapshot for repo, preprint, or package work.

The snapshot is not a release tool. It records the repository state, classifies
publication blockers, and creates a hash-backed inventory of files that are safe
or explicitly reviewed for release. Use it before arXiv updates, PyPI/conda
publishing, Zenodo/OSF deposits, or journal submission package assembly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import artifact_promotion
    import repo_map
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Could not import local agent modules from {SCRIPT_DIR}: {exc}") from exc


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


def build_snapshot(repo_root: Path, strict: bool = False) -> dict[str, object]:
    mapping = repo_map.build_map(repo_root)
    promotion = artifact_promotion.build_plan(repo_root)
    files = mapping["files"]
    tracked = [r for r in files if r["tracked"]]
    safe_tracked = [r for r in tracked if r["publish_policy"] == "commit"]
    review_tracked = [r for r in tracked if r["publish_policy"] == "review_before_commit"]
    blocked_tracked = [r for r in tracked if r["publish_policy"] == "do_not_commit"]
    dirty_status = run_git(repo_root, ["status", "--short"]) or ""
    tags = run_git(repo_root, ["tag", "--points-at", "HEAD"])
    blockers = [r["path"] for r in blocked_tracked]
    if strict:
        blockers.extend([r["path"] for r in review_tracked])
    ready = bool(mapping["repo"].get("is_git_repo")) and not dirty_status and not blockers
    return {
        "schema_version": "1.0",
        "repo": mapping["repo"],
        "detected_project": mapping["detected_project"],
        "git": {
            "branch": mapping["repo"].get("branch"),
            "head": mapping["repo"].get("head"),
            "tags_at_head": tags.splitlines() if tags else [],
            "dirty_status": dirty_status.splitlines(),
        },
        "strict": strict,
        "ready_to_release": ready,
        "summary": {
            "tracked_files": len(tracked),
            "safe_tracked_files": len(safe_tracked),
            "review_tracked_files": len(review_tracked),
            "blocked_tracked_files": len(blocked_tracked),
            "blockers": blockers,
            "promotion_action_counts": promotion["summary"]["action_counts"],
        },
        "safe_tracked_files": [
            {"path": r["path"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]} for r in safe_tracked
        ],
        "review_tracked_files": [
            {"path": r["path"], "sha256": r["sha256"], "size_bytes": r["size_bytes"], "category": r["category"]}
            for r in review_tracked
        ],
        "blocked_tracked_files": [
            {"path": r["path"], "sha256": r["sha256"], "size_bytes": r["size_bytes"], "category": r["category"]}
            for r in blocked_tracked
        ],
    }


def to_markdown(snapshot: dict[str, object]) -> str:
    repo = snapshot["repo"]
    summary = snapshot["summary"]
    lines = [
        f"# Release snapshot: {repo['name']}",
        "",
        f"- Ready to release: `{snapshot['ready_to_release']}`",
        f"- Strict mode: `{snapshot['strict']}`",
        f"- Branch: `{snapshot['git']['branch']}`",
        f"- HEAD: `{snapshot['git']['head']}`",
        f"- Tags at HEAD: {', '.join(snapshot['git']['tags_at_head']) or 'none'}",
        f"- Dirty status lines: {len(snapshot['git']['dirty_status'])}",
        "",
        "## Summary",
        "",
        f"- Tracked files: {summary['tracked_files']}",
        f"- Safe tracked files: {summary['safe_tracked_files']}",
        f"- Review tracked files: {summary['review_tracked_files']}",
        f"- Blocked tracked files: {summary['blocked_tracked_files']}",
        "",
    ]
    if summary["blockers"]:
        lines.extend(["## Blockers", ""])
        for path in summary["blockers"][:200]:
            lines.append(f"- `{path}`")
        lines.append("")
    if snapshot["git"]["dirty_status"]:
        lines.extend(["## Dirty status", ""])
        for line in snapshot["git"]["dirty_status"][:200]:
            lines.append(f"- `{line}`")
        lines.append("")
    lines.extend(["## Release rules", ""])
    lines.extend(
        [
            "- Do not publish if `ready_to_release` is false.",
            "- Use strict mode for journal/PyPI/Zenodo/OSF releases.",
            "- Review-tracked files require an explicit publication-policy allow rule or relocation.",
            "- Generated outputs cited in a manuscript require result manifests and stable hashes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    local_dir = repo_root / ".conductor" / "local"
    out_json = args.output_json or local_dir / "release_snapshot.json"
    out_md = args.output_md or local_dir / "release_snapshot.md"
    snapshot = build_snapshot(repo_root, strict=args.strict)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(to_markdown(snapshot), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0 if snapshot["ready_to_release"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
