#!/usr/bin/env python3
"""Plan a non-destructive migration from legacy v1-v5 conductor overlays to v6.

The planner never deletes or renames files. It detects legacy numbered tracks,
duplicate issue identifiers, old patch directories, and parallel perspective
surfaces, then writes an ordered migration plan under ``.conductor/local``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UpgradeAction:
    priority: int
    category: str
    action: str
    path: str
    rationale: str
    automatic: bool = False


def _legacy_tracks(repo: Path) -> list[Path]:
    root = repo / "conductor" / "tracks"
    if not root.exists():
        return []
    return sorted(path for path in root.glob("track_*.md") if path.is_file())


def _canonical_tracks(repo: Path) -> list[Path]:
    root = repo / "conductor" / "tracks"
    if not root.exists():
        return []
    return sorted(path for path in root.glob("C??_*.md") if path.is_file())


def _duplicate_issue_numbers(repo: Path) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    issue_root = repo / "issues"
    if not issue_root.exists():
        return buckets
    for path in issue_root.rglob("*.md"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            key = f"{path.parent.name}:{match.group(1)}"
            buckets.setdefault(key, []).append(str(path.relative_to(repo)))
    return {key: values for key, values in buckets.items() if len(values) > 1}


def _perspective_surfaces(repo: Path) -> list[str]:
    hits: list[str] = []
    for path in repo.rglob("*.py"):
        rel = str(path.relative_to(repo))
        lowered = rel.casefold()
        if any(token in lowered for token in ("perspective", "frontier", "value_of_information")):
            hits.append(rel)
    return sorted(hits)[:200]


def build_plan(repo: Path, pack_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    pack_root = pack_root.resolve()
    actions: list[UpgradeAction] = []
    legacy = _legacy_tracks(repo)
    canonical = _canonical_tracks(repo)
    duplicates = _duplicate_issue_numbers(repo)
    perspective = _perspective_surfaces(repo)

    if legacy:
        actions.append(
            UpgradeAction(
                10,
                "conductor",
                "map_legacy_tracks_to_canonical_registry_then_archive_locally",
                "conductor/tracks/track_*.md",
                "v6 replaces duplicate sequential numbering with stable semantic IDs C00-C12; preserve history but do not keep two active registries",
            )
        )
    if canonical:
        actions.append(
            UpgradeAction(
                20,
                "conductor",
                "merge_canonical_manifest_without_overwriting_local_evidence",
                "conductor/manifest.json",
                "an existing canonical surface requires a three-way merge rather than wholesale replacement",
            )
        )
    if duplicates:
        actions.append(
            UpgradeAction(
                30,
                "issues",
                "replace_numbered_markdown_source_of_truth_with_issues_backlog_json",
                "issues/",
                "duplicate issue numbers and titles make GitHub project creation non-idempotent",
            )
        )
    if perspective:
        actions.append(
            UpgradeAction(
                40,
                "method",
                "merge_method_contract_and_conformance_fixtures_into_existing_surface",
                "perspective implementation",
                "existing method code was detected; the v6 overlay is a conformance reference, not a parallel package",
            )
        )
    for patch_dir in (repo / "patches", repo / "patches_clean"):
        if patch_dir.exists():
            actions.append(
                UpgradeAction(
                    50,
                    "patches",
                    "treat_old_patches_as_local_history_or_release_artifacts",
                    str(patch_dir.relative_to(repo)),
                    "patch exports are not the live source of truth and should not accumulate indefinitely in production repos",
                )
            )
    actions.extend(
        [
            UpgradeAction(60, "integration", "run_pack_doctor", ".conductor/local/pack_doctor.md", "classify safe-add versus merge-required files against the actual worktree", True),
            UpgradeAction(70, "metadata", "run_metadata_and_repo_hygiene_audits", ".conductor/local/", "version, licence, citation, README, and root hygiene need truth before publication", True),
            UpgradeAction(80, "state", "initialise_resumable_track_state", ".conductor/local/track_state.json", "agents need dependency-aware state and evidence-backed completion", True),
        ]
    )
    return {
        "schema_version": "1.0",
        "target_pack_version": "6.0.0",
        "repo": str(repo),
        "pack_root": str(pack_root),
        "summary": {
            "legacy_tracks": len(legacy),
            "canonical_tracks": len(canonical),
            "duplicate_issue_groups": len(duplicates),
            "perspective_surfaces": len(perspective),
            "actions": len(actions),
        },
        "legacy_tracks": [str(path.relative_to(repo)) for path in legacy],
        "duplicate_issue_groups": duplicates,
        "perspective_surfaces": perspective,
        "actions": [asdict(action) for action in sorted(actions, key=lambda item: item.priority)],
        "rule": "No action in this report deletes, moves, or overwrites files automatically.",
    }


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Conductor v6 upgrade plan",
        "",
        f"- Legacy tracks: **{summary['legacy_tracks']}**",
        f"- Canonical tracks already present: **{summary['canonical_tracks']}**",
        f"- Duplicate issue groups: **{summary['duplicate_issue_groups']}**",
        f"- Existing perspective-related files: **{summary['perspective_surfaces']}**",
        "",
        "| Priority | Category | Action | Path | Automatic | Rationale |",
        "|---:|---|---|---|---|---|",
    ]
    for item in report["actions"]:
        lines.append(
            f"| {item['priority']} | {item['category']} | `{item['action']}` | `{item['path']}` | {item['automatic']} | {item['rationale']} |"
        )
    lines.extend(["", f"> {report['rule']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = build_plan(args.repo, args.pack_root)
    out = args.repo / ".conductor" / "local"
    out.mkdir(parents=True, exist_ok=True)
    (out / "upgrade_plan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "upgrade_plan.md").write_text(to_markdown(report), encoding="utf-8")
    print(out / "upgrade_plan.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
