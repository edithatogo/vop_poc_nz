#!/usr/bin/env python3
"""Create a non-destructive repository reorganisation plan.

This script proposes where local/generated/review-sensitive files should live.
It does not move files by default. It can emit a shell script containing commented
commands so a human or coding agent can apply them deliberately.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import artifact_promotion
    import repo_map
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Could not import local agent modules from {SCRIPT_DIR}: {exc}") from exc


@dataclass(frozen=True)
class ReorgProposal:
    path: str
    proposal: str
    priority: str
    tracked: bool
    suggested_destination: str | None
    shell_hint: str | None
    rationale: str


def proposed_destination(path: str, artifact_class: str) -> str | None:
    p = path.replace("\\", "/")
    if artifact_class in {"raw_or_private_data", "intermediate_or_binary_data"}:
        return f".conductor/local/data/{Path(p).name}"
    if artifact_class in {"generated_result_or_site", "generated_result_or_cache"}:
        return f".conductor/local/artifacts/{Path(p).name}"
    if artifact_class == "manuscript_or_review_material":
        return f".conductor/local/manuscripts/{Path(p).name}"
    if artifact_class in {"binary_document", "notebook", "fixture_needs_review"}:
        return f".conductor/local/review/{Path(p).name}"
    return None


def proposal_for_decision(decision: dict[str, object]) -> ReorgProposal | None:
    path = str(decision["path"])
    action = str(decision["action"])
    tracked = bool(decision["tracked"])
    artifact_class = str(decision["artifact_class"])
    target_state = str(decision["target_state"])
    destination = proposed_destination(path, artifact_class)

    if action == "commit_ok":
        return None
    if action == "commit_if_synthetic_or_redistributable":
        return ReorgProposal(
            path=path,
            proposal="confirm_fixture_publication",
            priority="medium",
            tracked=tracked,
            suggested_destination=path,
            shell_hint=None,
            rationale="Confirm the fixture is synthetic or redistributable and document provenance before committing.",
        )
    if tracked and action == "untrack_or_move_before_push":
        shell = f"git rm --cached -- {shlex.quote(path)}  # then move/copy to {shlex.quote(destination or '.conductor/local/review/') } if needed"
        return ReorgProposal(
            path=path,
            proposal="untrack_local_only_file",
            priority="high",
            tracked=tracked,
            suggested_destination=destination,
            shell_hint=shell,
            rationale="Tracked local-only/generated material should be removed from the Git index before publication.",
        )
    if target_state in {"external_artifact", "manifest_backed", "local_reviewed"}:
        shell = None
        if destination and not path.startswith(".conductor/local/"):
            shell = f"mkdir -p {shlex.quote(str(Path(destination).parent))} && git mv --force -- {shlex.quote(path)} {shlex.quote(destination)}  # only if this file truly belongs locally"
        return ReorgProposal(
            path=path,
            proposal="move_or_manifest_local_artifact",
            priority="medium" if not tracked else "high",
            tracked=tracked,
            suggested_destination=destination,
            shell_hint=shell,
            rationale="Keep local artifacts out of the public package unless explicitly promoted with a manifest.",
        )
    return ReorgProposal(
        path=path,
        proposal="manual_review",
        priority="low",
        tracked=tracked,
        suggested_destination=destination,
        shell_hint=None,
        rationale="No automatic reorganisation rule matched; inspect manually.",
    )


def build_plan(repo_root: Path) -> dict[str, object]:
    promotion = artifact_promotion.build_plan(repo_root)
    proposals = [proposal_for_decision(decision) for decision in promotion["decisions"]]
    proposals = [proposal for proposal in proposals if proposal is not None]
    priority_counts: dict[str, int] = {}
    proposal_counts: dict[str, int] = {}
    for proposal in proposals:
        priority_counts[proposal.priority] = priority_counts.get(proposal.priority, 0) + 1
        proposal_counts[proposal.proposal] = proposal_counts.get(proposal.proposal, 0) + 1
    return {
        "schema_version": "1.0",
        "repo": promotion["repo"],
        "detected_project": promotion["detected_project"],
        "summary": {
            "total_proposals": len(proposals),
            "priority_counts": priority_counts,
            "proposal_counts": proposal_counts,
            "high_priority_paths": [p.path for p in proposals if p.priority == "high"],
        },
        "proposals": [asdict(proposal) for proposal in proposals],
    }


def to_markdown(plan: dict[str, object]) -> str:
    repo = plan["repo"]
    summary = plan["summary"]
    lines = [
        f"# Repository reorganisation plan: {repo['name']}",
        "",
        "This is a non-destructive plan. Do not apply shell hints without reviewing the file contents and Git diff.",
        "",
        "## Summary",
        "",
        f"- Total proposals: {summary['total_proposals']}",
        "",
        "### Priority counts",
        "",
    ]
    for key, value in sorted(summary["priority_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "### Proposal counts", ""])
    for key, value in sorted(summary["proposal_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    high = summary.get("high_priority_paths", [])
    if high:
        lines.extend(["", "## High-priority paths", ""])
        for path in high[:200]:
            lines.append(f"- `{path}`")
    lines.extend(["", "## Proposals", ""])
    lines.append("| Priority | Proposal | Path | Tracked | Suggested destination | Shell hint | Rationale |")
    lines.append("|---|---|---|---:|---|---|---|")
    for proposal in plan["proposals"][:500]:
        lines.append(
            "| `{priority}` | `{proposal}` | `{path}` | {tracked} | {suggested_destination} | `{shell_hint}` | {rationale} |".format(
                **{
                    **proposal,
                    "suggested_destination": proposal.get("suggested_destination") or "",
                    "shell_hint": proposal.get("shell_hint") or "",
                }
            )
        )
    lines.extend(
        [
            "",
            "## Agent rules",
            "",
            "- Prefer `git rm --cached` over deleting local files when the content should remain available locally.",
            "- Keep raw empirical sources and reviewer letters outside the public repo unless there is an explicit publication decision.",
            "- If a generated figure/table is committed, also commit the result manifest and generation script.",
            "- Make reorganisation a separate commit from scientific code changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def to_shell(plan: dict[str, object]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Review before running. Commands are commented intentionally.",
        "",
    ]
    for proposal in plan["proposals"]:
        hint = proposal.get("shell_hint")
        if hint:
            lines.append(f"# {hint}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--write-shell", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    local_dir = repo_root / ".conductor" / "local"
    out_json = args.output_json or local_dir / "reorg_plan.json"
    out_md = args.output_md or local_dir / "reorg_plan.md"
    plan = build_plan(repo_root)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(to_markdown(plan), encoding="utf-8")
    if args.write_shell:
        shell = local_dir / "reorg_plan.reviewed_commands.sh"
        shell.write_text(to_shell(plan), encoding="utf-8")
        print(f"Wrote {shell}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
