#!/usr/bin/env python3
"""Run the v6 map-first conductor gates and preserve a machine-readable audit.

The command is intentionally conservative. It does not edit source files. Some
strict publication gates are expected to fail while work is in progress; use
``--keep-going`` to collect the complete report instead of stopping at the first
blocker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class GateSpec:
    name: str
    script: str
    args: tuple[str, ...] = ()
    target: str = "repo"  # repo, pack, or context


GATES: tuple[GateSpec, ...] = (
    GateSpec("pack_doctor", "pack_doctor.py", target="repo_with_pack"),
    GateSpec("conductor_registry", "conductor_registry.py", target="pack"),
    GateSpec("issue_registry", "issue_registry.py", ("--check",), target="pack"),
    GateSpec("pack_self_check", "pack_self_check.py", target="pack"),
    GateSpec("track_state", "track_state.py", ("--init",), target="repo_with_pack"),
    GateSpec("conductor_status", "conductor_status.py", target="repo_with_pack"),
    GateSpec("repo_map", "repo_map.py"),
    GateSpec("git_safety", "git_safety.py"),
    GateSpec("upgrade_plan", "upgrade_plan.py", target="repo_with_pack"),
    GateSpec("repo_hygiene", "repo_hygiene.py"),
    GateSpec("metadata_consistency", "metadata_consistency.py", ("--strict",)),
    GateSpec("artifact_promotion", "artifact_promotion.py"),
    GateSpec("reorg_plan", "reorg_plan.py"),
    GateSpec("manuscript_reconcile", "manuscript_reconcile.py", ("--strict",)),
    GateSpec("evidence_audit", "evidence_audit.py", ("--strict",)),
    GateSpec("evidence_ledger_audit", "evidence_ledger_audit.py", ("--strict",)),
    GateSpec("reviewer_matrix", "reviewer_matrix.py"),
    GateSpec("concept_budget", "concept_budget.py", ("--strict",)),
    GateSpec("manuscript_output_audit", "manuscript_output_audit.py", ("--strict",)),
    GateSpec("governance_harness", "governance_harness.py", ("--strict",)),
    GateSpec("import_boundary", "import_boundary.py", ("--strict",)),
    GateSpec("issue_hygiene", "issue_hygiene.py", ("--strict",)),
    GateSpec("publication_gate", "publication_gate.py", ("--strict",)),
    GateSpec("release_snapshot", "release_snapshot.py", ("--strict",)),
    GateSpec("context_pack", "context_pack.py"),
    GateSpec("prompt_series", "prompt_series.py", target="context"),
)


@dataclass(frozen=True)
class GateResult:
    gate: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    missing: bool = False


def command_for(spec: GateSpec, script_dir: Path, repo: Path, pack_root: Path) -> list[str]:
    script = script_dir / spec.script
    base = [sys.executable, str(script)]
    if spec.target == "pack":
        return [*base, str(pack_root), *spec.args]
    if spec.target == "repo_with_pack":
        return [*base, str(repo), "--pack-root", str(pack_root), *spec.args]
    if spec.target == "context":
        context_path = repo / ".conductor" / "local" / "AGENT_CONTEXT.json"
        return [*base, str(context_path), *spec.args]
    return [*base, str(repo), *spec.args]


def run_gate(spec: GateSpec, script_dir: Path, repo: Path, pack_root: Path) -> GateResult:
    script = script_dir / spec.script
    command = command_for(spec, script_dir, repo, pack_root)
    if not script.exists():
        return GateResult(spec.name, command, 127, "", f"missing script: {script}", True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return GateResult(
        spec.name,
        command,
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def to_markdown(results: list[GateResult], *, repo: Path, pack_root: Path) -> str:
    passed = sum(result.returncode == 0 for result in results)
    lines = [
        "# Local conductor gate run",
        "",
        f"- Repository: `{repo}`",
        f"- Pack root: `{pack_root}`",
        f"- Passed: **{passed} / {len(results)}**",
        "",
        "| Gate | Return code | Command | Output |",
        "|---|---:|---|---|",
    ]
    for result in results:
        output = (result.stdout or result.stderr or "").replace("|", "\\|").replace("\n", "<br>")[:900]
        command = " ".join(result.command).replace("|", "\\|")
        lines.append(f"| `{result.gate}` | {result.returncode} | `{command}` | {output} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A non-zero strict gate is a recorded blocker, not permission to bypass it.",
            "- Merge-required integration items must be resolved in the live architecture.",
            "- The final release/submission decision should cite concrete evidence in `track_state.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    pack_root = args.pack_root.resolve()
    script_dir = Path(__file__).resolve().parent
    results: list[GateResult] = []
    for spec in GATES:
        result = run_gate(spec, script_dir, repo, pack_root)
        results.append(result)
        if result.returncode != 0 and not args.keep_going:
            break
    local_dir = repo / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0",
        "repo": str(repo),
        "pack_root": str(pack_root),
        "summary": {
            "passed": sum(result.returncode == 0 for result in results),
            "failed": sum(result.returncode != 0 for result in results),
            "total": len(results),
            "all_passed": all(result.returncode == 0 for result in results),
        },
        "results": [asdict(result) for result in results],
    }
    (local_dir / "local_gate_run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (local_dir / "local_gate_run.md").write_text(to_markdown(results, repo=repo, pack_root=pack_root), encoding="utf-8")
    print(local_dir / "local_gate_run.md")
    return 0 if payload["summary"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
