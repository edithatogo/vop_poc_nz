#!/usr/bin/env python3
"""Plan selective integration of a conductor overlay into a mapped repo.

The script never overwrites existing files. It creates an integration plan that
identifies safe additions, exact matches, conflicts requiring review, and local
repo-organisation work. Use --apply-safe only after reviewing the Markdown plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import repo_map
    import git_safety
except ImportError as exc:  # pragma: no cover - exercised only in broken installs
    raise SystemExit(f"Could not import repo_map.py from {SCRIPT_DIR}: {exc}") from exc


@dataclass(frozen=True)
class OverlayAction:
    action: str
    source: str
    destination: str
    status: str
    publish_policy: str
    reason: str


@dataclass(frozen=True)
class ReorgAction:
    action: str
    path: str
    status: str
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_kind(root: Path) -> str:
    detected = repo_map.detect_project(root)
    return str(detected.get("kind", "unknown"))


def default_overlay_roots(pack_root: Path, kind: str) -> list[Path]:
    roots: list[Path] = []
    if kind != "unknown":
        candidate = pack_root / "overlays" / kind
        if candidate.exists():
            roots.append(candidate)
    shared = pack_root / "overlays" / "shared_local_agent"
    if shared.exists():
        roots.append(shared)
    return roots


def iter_overlay_files(overlay_root: Path):
    for path in sorted(overlay_root.rglob("*")):
        if path.is_file():
            yield path


def plan_overlay(repo_root: Path, overlay_roots: list[Path]) -> list[OverlayAction]:
    actions: list[OverlayAction] = []
    for overlay_root in overlay_roots:
        for src in iter_overlay_files(overlay_root):
            rel = src.relative_to(overlay_root).as_posix()
            dst = repo_root / rel
            if not dst.exists():
                status = "safe_add"
                reason = "destination does not exist"
            else:
                if sha256_file(src) == sha256_file(dst):
                    status = "already_same"
                    reason = "destination already contains identical content"
                else:
                    status = "conflict_review"
                    reason = "destination exists with different content; do not overwrite automatically"
            category, publish_policy, class_reasons = repo_map.classify(rel)
            actions.append(
                OverlayAction(
                    action="copy_overlay_file",
                    source=str(src),
                    destination=rel,
                    status=status,
                    publish_policy=publish_policy,
                    reason=f"{reason}; classified as {category}: {'; '.join(class_reasons)}",
                )
            )
    return actions


def plan_reorg(mapping: dict[str, object]) -> list[ReorgAction]:
    actions: list[ReorgAction] = []
    for record in mapping["files"]:
        path = str(record["path"])
        tracked = bool(record["tracked"])
        policy = str(record["publish_policy"])
        category = str(record["category"])
        if tracked and policy == "do_not_commit":
            actions.append(
                ReorgAction(
                    action="remove_from_git_or_move_to_local_workspace",
                    path=path,
                    status="requires_human_review",
                    reason=f"tracked file is classified as {category}/{policy}",
                )
            )
        elif tracked and policy == "review_before_commit":
            actions.append(
                ReorgAction(
                    action="confirm_publication_boundary",
                    path=path,
                    status="requires_human_review",
                    reason="tracked file may be legitimate but needs explicit publication-policy decision",
                )
            )
        elif (not tracked) and policy == "commit":
            actions.append(
                ReorgAction(
                    action="consider_adding_public_source_file",
                    path=path,
                    status="optional",
                    reason="untracked file appears to be public source/docs/test material",
                )
            )
    return actions


def build_plan(repo_root: Path, pack_root: Path, overlay_roots: list[Path] | None = None) -> dict[str, object]:
    repo_root = repo_root.resolve()
    pack_root = pack_root.resolve()
    mapping = repo_map.build_map(repo_root)
    kind = str(mapping["detected_project"]["kind"])
    roots = overlay_roots or default_overlay_roots(pack_root, kind)
    overlay_actions = plan_overlay(repo_root, roots)
    reorg_actions = plan_reorg(mapping)
    return {
        "schema_version": "1.0",
        "repo": mapping["repo"],
        "detected_project": mapping["detected_project"],
        "overlay_roots": [str(path) for path in roots],
        "overlay_actions": [asdict(action) for action in overlay_actions],
        "reorg_actions": [asdict(action) for action in reorg_actions],
        "summary": {
            "safe_add": sum(1 for a in overlay_actions if a.status == "safe_add"),
            "already_same": sum(1 for a in overlay_actions if a.status == "already_same"),
            "conflict_review": sum(1 for a in overlay_actions if a.status == "conflict_review"),
            "reorg_review": sum(1 for a in reorg_actions if a.status == "requires_human_review"),
        },
    }


def plan_to_markdown(plan: dict[str, object]) -> str:
    repo = plan["repo"]
    detected = plan["detected_project"]
    summary = plan["summary"]
    lines = [
        f"# Integration plan: {repo['name']}",
        "",
        f"- Root: `{repo['root']}`",
        f"- Detected project: `{detected['kind']}`",
        f"- Safe overlay additions: {summary['safe_add']}",
        f"- Already identical: {summary['already_same']}",
        f"- Conflicts needing review: {summary['conflict_review']}",
        f"- Repo-organisation items needing review: {summary['reorg_review']}",
        "",
        "## Overlay actions",
        "",
        "| Status | Destination | Policy | Reason |",
        "|---|---|---:|---|",
    ]
    for action in plan["overlay_actions"]:
        lines.append(
            f"| `{action['status']}` | `{action['destination']}` | `{action['publish_policy']}` | {action['reason']} |"
        )
    lines.extend(["", "## Repository organisation actions", ""])
    lines.append("| Status | Action | Path | Reason |")
    lines.append("|---|---|---|---|")
    for action in plan["reorg_actions"]:
        lines.append(
            f"| `{action['status']}` | `{action['action']}` | `{action['path']}` | {action['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Agent rules",
            "",
            "- Apply only `safe_add` actions automatically.",
            "- Never overwrite `conflict_review` files without a human-reviewed diff.",
            "- Do not push files classified as `do_not_commit`.",
            "- Convert `review_before_commit` files into explicit policy decisions before pushing.",
        ]
    )
    return "\n".join(lines) + "\n"


def apply_safe(plan: dict[str, object], repo_root: Path) -> list[str]:
    copied: list[str] = []
    for action in plan["overlay_actions"]:
        if action["status"] != "safe_add":
            continue
        src = Path(action["source"])
        dst = repo_root / str(action["destination"])
        if dst.exists():
            raise RuntimeError(f"Refusing to overwrite {dst}; regenerate the integration plan")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(action["destination"]))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overlay-root", action="append", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--apply-safe", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-default-branch", action="store_true")
    parser.add_argument("--allow-detached", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    overlay_roots = [p.resolve() for p in args.overlay_root] if args.overlay_root else None
    plan = build_plan(repo_root, args.pack_root, overlay_roots)
    default_out = repo_root / ".conductor" / "local"
    out_json = args.output_json or default_out / "integration_plan.json"
    out_md = args.output_md or default_out / "integration_plan.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(plan_to_markdown(plan), encoding="utf-8")

    if args.apply_safe:
        git_safety.require_safe(
            repo_root,
            allow_dirty=args.allow_dirty,
            allow_default_branch=args.allow_default_branch,
            allow_detached=args.allow_detached,
        )
        copied = apply_safe(plan, repo_root)
        print(f"Copied {len(copied)} safe overlay files")
        for path in copied:
            print(f"  {path}")
    else:
        print(f"Wrote {out_json}")
        print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
