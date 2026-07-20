#!/usr/bin/env python3
"""Render a state-aware v6 local-agent prompt series from an agent context pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPTS = [
    ("P00 — Intake and map", "Read `pack_doctor.md`, `upgrade_plan.md`, `repo_map.md`, `repo_hygiene.md`, and `metadata_consistency.md`. Do not edit before the live architecture and local/public boundary are understood."),
    ("P01 — Select one canonical track", "Read `conductor_status.md`; choose one dependency-ready `C##` track and record `in_progress` state, intended files, and completion evidence."),
    ("P02 — Lock method semantics", "For EVoP/PAF changes, verify the versioned contract, tie policy, direction, decision rule, exact frontier, conformance fixtures, and Monte Carlo assurance before coding."),
    ("P03 — Merge implementation", "Apply only `safe_add` files automatically. Merge every `merge_required` item into the existing API, CLI, registry, tests, and docs; do not create parallel surfaces."),
    ("P04 — Evidence and validation", "Resolve evidence-ledger errors, generate model cards, run internal model checks, and document external-validity limits and numerical convergence."),
    ("P05 — Artifacts and pipeline", "Make every public result manifest-backed with inputs, hashes, seeds, software/method versions, and promotion state. Keep private and generated working material local."),
    ("P06 — Manuscript and scope", "Reconcile every figure/table/number to current outputs and enforce the three-concept budget: directional EVoP, PAF, and regime discovery."),
    ("P07 — Release and public boundary", "Resolve version/licence/citation/docs/package-index truth, run publication and release gates, and inspect the artifact ledger before any push or submission."),
    ("P08 — Closeout", "Run the full live-repo suite and all local gates. Complete the track only with evidence and commit references; create canonical issues for residual blockers."),
]


def render_prompt_series(context: dict[str, object], track: str | None = None) -> str:
    repo = context.get("repo", {})
    blockers = context.get("blockers", [])
    active = track or context.get("active_track") or "select from conductor_status.md"
    lines = [
        f"# Local agent prompt series: {repo.get('name', 'repo')}",
        "",
        "## Map-first invariant",
        "",
        "Map first; merge into the live architecture; promote artifacts explicitly; complete tracks only with evidence.",
        "",
        f"Active track: `{active}`",
        "",
        "## Current blockers",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None detected by generated reports.")
    lines.extend(
        [
            "",
            "## Stop conditions",
            "",
            "Stop and record a blocker if repository identity is ambiguous, a merge would overwrite live work, method semantics lack tests, evidence provenance is unresolved for a public claim, or publication gates would be bypassed.",
            "",
            "## Prompt sequence",
            "",
        ]
    )
    for title, body in PROMPTS:
        lines.extend([f"### {title}", "", body, ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context_json", type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--track", default=None)
    args = parser.parse_args()
    context = json.loads(args.context_json.read_text(encoding="utf-8"))
    output = render_prompt_series(context, track=args.track)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
