#!/usr/bin/env python3
"""Build a compact agent context pack from local conductor reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPORTS = {
    "repo_map": "repo_map.json",
    "git_safety": "git_safety.json",
    "upgrade_plan": "upgrade_plan.json",
    "integration_plan": "integration_plan.json",
    "artifact_promotion": "artifact_promotion_plan.json",
    "reorg_plan": "reorg_plan.json",
    "release_snapshot": "release_snapshot.json",
    "manuscript_reconciliation": "manuscript_reconciliation.json",
    "evidence_audit": "evidence_audit.json",
    "concept_budget": "concept_budget.json",
    "import_boundary": "import_boundary.json",
    "reviewer_response_matrix": "reviewer_response_matrix.json",
    "pack_doctor": "pack_doctor.json",
    "conductor_registry": "conductor_registry.json",
    "conductor_status": "conductor_status.json",
    "repo_hygiene": "repo_hygiene.json",
    "metadata_consistency": "metadata_consistency.json",
}


def run_git(root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    except Exception:
        return None


def load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarise_report(name: str, data: object | None) -> dict[str, object]:
    if not isinstance(data, dict):
        return {"available": False}
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    if name == "git_safety":
        summary = {
            "safe": data.get("safe"),
            "branch": data.get("branch"),
            "tracked_dirty": data.get("tracked_dirty"),
            "staged_dirty": data.get("staged_dirty"),
            "appears_default_branch": data.get("appears_default_branch"),
            "blockers": data.get("blockers", []),
        }
    return {"available": True, "summary": summary}


def build_context(repo_root: Path, active_track: str | None = None) -> dict[str, object]:
    repo_root = repo_root.resolve()
    local_dir = repo_root / ".conductor" / "local"
    reports = {name: summarise_report(name, load_json(local_dir / filename)) for name, filename in REPORTS.items()}
    blockers: list[str] = []
    release = reports.get("release_snapshot", {}).get("summary", {})
    if isinstance(release, dict) and release.get("blockers"):
        blockers.extend(str(item) for item in release.get("blockers", []))
    manuscript = reports.get("manuscript_reconciliation", {}).get("summary", {})
    if isinstance(manuscript, dict) and manuscript.get("unresolved_references", 0):
        blockers.append("unresolved manuscript figure/table references")
    evidence = reports.get("evidence_audit", {}).get("summary", {})
    if isinstance(evidence, dict) and not evidence.get("ready_for_empirical_claims", True):
        blockers.append("evidence ledger audit not ready for empirical claims")
    boundary = reports.get("import_boundary", {}).get("summary", {})
    if isinstance(boundary, dict) and not boundary.get("boundary_pass", True):
        blockers.append("cross-repo import boundary violations")
    budget = reports.get("concept_budget", {}).get("summary", {})
    if isinstance(budget, dict) and not budget.get("scope_budget_pass", True):
        blockers.append("concept scope budget failed")
    safety = reports.get("git_safety", {}).get("summary", {})
    if isinstance(safety, dict) and safety and not safety.get("safe", True):
        blockers.append("Git worktree is unsafe for mutating integration; use a clean named non-default branch/worktree")
    doctor = reports.get("pack_doctor", {}).get("summary", {})
    if isinstance(doctor, dict) and doctor.get("merge_required", 0):
        blockers.append(f"{doctor.get('merge_required')} pack items require architecture-aware merging")
    metadata = reports.get("metadata_consistency", {}).get("summary", {})
    if isinstance(metadata, dict) and not metadata.get("consistent", True):
        blockers.append("version, licence, citation, or README metadata are inconsistent")
    registry = reports.get("conductor_registry", {}).get("summary", {})
    if isinstance(registry, dict) and not registry.get("valid", True):
        blockers.append("canonical conductor registry is invalid")
    notable_files = [str(path.relative_to(repo_root)) for path in repo_root.rglob("*") if path.is_file() and path.name in {"results.pkl", "results.parquet", "README.md"}][:25]
    return {
        "schema_version": "1.0",
        "repo_root": str(repo_root),
        "active_track": active_track,
        "notable_files": notable_files,
        "repo": {
            "root": str(repo_root),
            "name": repo_root.name,
            "branch": run_git(repo_root, ["branch", "--show-current"]),
            "head": run_git(repo_root, ["rev-parse", "--short", "HEAD"]),
            "dirty_status": run_git(repo_root, ["status", "--short"]),
        },
        "reports": reports,
        "blockers": sorted(set(blockers)),
        "recommended_next_commands": [
            "python scripts/pack_doctor.py . --pack-root /path/to/pack",
            "python scripts/conductor_registry.py /path/to/pack",
            "python scripts/conductor_status.py . --pack-root /path/to/pack",
            "python scripts/git_safety.py . --strict",
            "python scripts/metadata_consistency.py . --strict",
            "python scripts/repo_hygiene.py .",
            "python scripts/manuscript_reconcile.py .",
            "python scripts/evidence_audit.py . --strict",
            "python scripts/concept_budget.py . --strict",
            "python scripts/import_boundary.py . --strict",
            "python scripts/publication_gate.py . --strict",
            "python scripts/release_snapshot.py . --strict",
            "python scripts/run_all_local_gates.py . --pack-root /path/to/pack --keep-going",
        ],
    }


def to_markdown(context: dict[str, object]) -> str:
    repo = context["repo"]
    lines = [f"# Agent context pack: {repo['name']}", "", "## Repo", ""]
    for key in ("root", "branch", "head"):
        lines.append(f"- `{key}`: `{repo.get(key)}`")
    lines.append(f"- `dirty_status`: `{repo.get('dirty_status') or ''}`")
    lines.extend(["", "## Reports", ""])
    for name, report in context["reports"].items():
        lines.append(f"- `{name}`: available=`{report.get('available')}` summary=`{report.get('summary', {})}`")
    lines.extend(["", "## Notable files", ""])
    notable = context.get("notable_files", [])
    if notable:
        for item in notable:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Blockers", ""])
    if context["blockers"]:
        for blocker in context["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None detected by local conductor reports.")
    lines.extend(["", "## Recommended next commands", ""])
    for command in context["recommended_next_commands"]:
        lines.append(f"```bash\n{command}\n```")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--active-track", default=None)
    args = parser.parse_args()
    context = build_context(args.repo, active_track=args.active_track)
    out_dir = args.repo / ".conductor" / "local"
    output_json = args.output_json or out_dir / "AGENT_CONTEXT.json"
    output_md = args.output_md or out_dir / "AGENT_CONTEXT.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(to_markdown(context), encoding="utf-8")
    print(f"Agent context written: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
