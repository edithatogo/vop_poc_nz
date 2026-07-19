#!/usr/bin/env python3
"""Plan artifact promotion after mapping a local repository.

The artifact promotion planner is deliberately conservative. It does not move,
copy, delete, or publish files. It reads the repository map produced by
``repo_map.py`` and assigns each file to a publication lifecycle state so a local
agent can decide what should remain local, what needs a manifest, what should be
externalised to OSF/Zenodo, and what is safe to commit.

Lifecycle states used here:

``local_scratch``
    Local-only, generated, private, or otherwise unsuitable for GitHub.

``local_reviewed``
    Potentially publishable but needs explicit human/source review first.

``manifest_backed``
    Public metadata/provenance that should be backed by a schema, manifest, or
    evidence ledger.

``external_artifact``
    Outputs or source materials that should usually be stored in OSF/Zenodo or a
    private data store rather than committed to the package repository.

``public_fixture``
    Small, synthetic, permission-safe, deterministic, or schema-backed examples
    suitable for tests/tutorials.

``public_source``
    Source code, tests, schemas, conductor tracks, and public documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import repo_map
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Could not import repo_map.py from {SCRIPT_DIR}: {exc}") from exc

BINARY_OR_SUBMISSION_SUFFIXES = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".zip",
    ".tar", ".gz", ".7z", ".png", ".jpg", ".jpeg", ".webp", ".tiff",
}

DATA_OUTPUT_SUFFIXES = {
    ".csv", ".tsv", ".parquet", ".arrow", ".feather", ".pkl", ".pickle",
    ".sqlite", ".sqlite3", ".db", ".jsonl", ".ndjson",
}

MANIFEST_HINTS = (
    "result_manifest",
    "evidence_ledger",
    "case_contract",
    "perspective_manifest",
    "model_card",
    "mcda_feature",
    "net_benefit_tensor",
)

EXTERNAL_PATH_HINTS = (
    "manuscripts/submissions/",
    "reviewer_letters/",
    "osf/",
    "zenodo/",
    "artifacts/release/",
)

PUBLIC_FIXTURE_PREFIXES = (
    "examples/",
    "tests/fixtures/",
    "tests/data/",
    "docs/examples/",
)

PUBLIC_SOURCE_PREFIXES = (
    "src/",
    "tests/",
    "docs/",
    "conductor/",
    "schemas/",
    "scripts/",
    "adr/",
    "issues/",
    "prompts/",
    ".github/",
)


@dataclass(frozen=True)
class ArtifactDecision:
    path: str
    tracked: bool
    category: str
    publish_policy: str
    lifecycle_state: str
    recommended_action: str
    destination_hint: str
    reasons: list[str]


def _lower(path: str) -> str:
    return path.replace("\\", "/").lower()


def _suffix(path: str) -> str:
    return Path(path).suffix.lower()


def classify_artifact(record: dict[str, object]) -> ArtifactDecision:
    """Classify a mapped file into an artifact promotion lifecycle state."""
    path = str(record["path"])
    lower = _lower(path)
    suffix = _suffix(path)
    category = str(record.get("category", "needs_review"))
    publish_policy = str(record.get("publish_policy", "review_before_commit"))
    tracked = bool(record.get("tracked", False))
    reasons = list(record.get("reasons", []))

    if any(hint in lower for hint in MANIFEST_HINTS):
        reasons.append("name indicates a schema-backed manifest/ledger/model-card artifact")
        return ArtifactDecision(
            path=path,
            tracked=tracked,
            category=category,
            publish_policy=publish_policy,
            lifecycle_state="manifest_backed",
            recommended_action="validate_schema_then_commit_or_reference",
            destination_hint="schemas/, examples/, docs/model_cards/, or .conductor/manifests/",
            reasons=reasons,
        )

    if publish_policy == "do_not_commit":
        if any(lower.startswith(prefix) for prefix in EXTERNAL_PATH_HINTS) or suffix in BINARY_OR_SUBMISSION_SUFFIXES:
            reasons.append("not suitable for GitHub; keep local or register as an external artifact")
            action = "untrack_or_externalise" if tracked else "keep_local_or_externalise"
            return ArtifactDecision(
                path=path,
                tracked=tracked,
                category=category,
                publish_policy=publish_policy,
                lifecycle_state="external_artifact",
                recommended_action=action,
                destination_hint="OSF/Zenodo/private data store with manifest hash only in GitHub",
                reasons=reasons,
            )
        reasons.append("local-only/generated file should not be in the public repo")
        action = "untrack_or_move_to_local_workspace" if tracked else "keep_ignored_local"
        return ArtifactDecision(
            path=path,
            tracked=tracked,
            category=category,
            publish_policy=publish_policy,
            lifecycle_state="local_scratch",
            recommended_action=action,
            destination_hint=".conductor/local/, results/local/, outputs/local/, artifacts/local/",
            reasons=reasons,
        )

    if publish_policy == "review_before_commit":
        if any(lower.startswith(prefix) for prefix in EXTERNAL_PATH_HINTS) or suffix in BINARY_OR_SUBMISSION_SUFFIXES:
            reasons.append("binary/submission/publication-adjacent artifact requires explicit publication review")
            return ArtifactDecision(
                path=path,
                tracked=tracked,
                category=category,
                publish_policy=publish_policy,
                lifecycle_state="external_artifact",
                recommended_action="review_license_privacy_then_externalise_or_document",
                destination_hint="external artifact store; commit only manifest, thumbnail, or derived fixture if permitted",
                reasons=reasons,
            )
        if suffix in DATA_OUTPUT_SUFFIXES or lower.startswith(("data/", "results/", "outputs/", "artifacts/")):
            reasons.append("data/output artifact needs a manifest and provenance before publication")
            return ArtifactDecision(
                path=path,
                tracked=tracked,
                category=category,
                publish_policy=publish_policy,
                lifecycle_state="local_reviewed",
                recommended_action="create_result_or_evidence_manifest_before_commit",
                destination_hint=".conductor/manifests/ or examples/fixtures/ after review",
                reasons=reasons,
            )
        reasons.append("review-required file; inspect before adding to public package")
        return ArtifactDecision(
            path=path,
            tracked=tracked,
            category=category,
            publish_policy=publish_policy,
            lifecycle_state="local_reviewed",
            recommended_action="human_review_before_commit",
            destination_hint="public path only after review; otherwise .conductor/local/",
            reasons=reasons,
        )

    if any(lower.startswith(prefix) for prefix in PUBLIC_FIXTURE_PREFIXES):
        reasons.append("small public fixture/example path; verify it is synthetic or redistributable")
        return ArtifactDecision(
            path=path,
            tracked=tracked,
            category=category,
            publish_policy=publish_policy,
            lifecycle_state="public_fixture",
            recommended_action="commit_if_synthetic_schema_backed_or_redistributable",
            destination_hint="examples/, tests/fixtures/, docs/examples/",
            reasons=reasons,
        )

    if any(lower.startswith(prefix) for prefix in PUBLIC_SOURCE_PREFIXES) or Path(path).name in repo_map.PUBLIC_ROOT_FILES:
        reasons.append("public source/configuration/documentation path")
        return ArtifactDecision(
            path=path,
            tracked=tracked,
            category=category,
            publish_policy=publish_policy,
            lifecycle_state="public_source",
            recommended_action="commit_after_tests_and_review",
            destination_hint="GitHub repository",
            reasons=reasons,
        )

    reasons.append("no specific artifact promotion rule matched; review conservatively")
    return ArtifactDecision(
        path=path,
        tracked=tracked,
        category=category,
        publish_policy=publish_policy,
        lifecycle_state="local_reviewed",
        recommended_action="human_review_before_commit",
        destination_hint="GitHub only after explicit decision",
        reasons=reasons,
    )


def _target_state_for(decision: ArtifactDecision) -> str:
    if decision.lifecycle_state == "local_reviewed" and decision.recommended_action == "create_result_or_evidence_manifest_before_commit":
        return "manifest_backed"
    if decision.lifecycle_state == "external_artifact":
        return "external_artifact"
    if decision.lifecycle_state == "local_scratch":
        return "local_scratch"
    if decision.lifecycle_state in {"public_fixture", "public_source"}:
        return "public_release"
    return decision.lifecycle_state


def _current_state_for(decision: ArtifactDecision) -> str:
    if decision.lifecycle_state in {"public_fixture", "public_source"}:
        return "public_release"
    return decision.lifecycle_state


def _action_for(decision: ArtifactDecision) -> str:
    if decision.lifecycle_state == "local_scratch" and not decision.tracked:
        return "keep_local_ignored"
    if decision.lifecycle_state == "local_scratch" and decision.tracked:
        return "untrack_or_move_before_push"
    if decision.lifecycle_state == "external_artifact" and decision.tracked:
        return "externalise_or_untrack_before_push"
    if decision.lifecycle_state == "external_artifact":
        return "keep_external_or_manifest_hash_only"
    if decision.lifecycle_state == "public_source":
        return "commit_ok"
    if decision.lifecycle_state == "public_fixture":
        return "commit_if_synthetic_or_redistributable"
    if _target_state_for(decision) == "manifest_backed":
        return "create_manifest_before_publication"
    return decision.recommended_action


def build_promotion_plan(mapping: dict[str, object]) -> dict[str, object]:
    """Build a compact artifact-promotion plan from an existing repo map.

    This is the stable local-agent contract used by tests and prompt-series
    tooling. It is intentionally simpler than the full artifact plan.
    """
    decisions = [classify_artifact(record) for record in mapping["files"]]
    records = []
    target_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for decision in decisions:
        current_state = _current_state_for(decision)
        target_state = _target_state_for(decision)
        action = _action_for(decision)
        approval_required = target_state not in {"public_release", "local_scratch"} or decision.publish_policy != "commit"
        record = {
            "path": decision.path,
            "tracked": decision.tracked,
            "category": decision.category,
            "publish_policy": decision.publish_policy,
            "current_state": current_state,
            "recommended_target_state": target_state,
            "approval_required": approval_required,
            "recommended_action": action,
            "destination_hint": decision.destination_hint,
            "reasons": decision.reasons,
        }
        target_counts[target_state] += 1
        action_counts[action] += 1
        records.append(record)
    return {
        "schema_version": "1.0",
        "repo": mapping["repo"],
        "detected_project": mapping["detected_project"],
        "summary": {
            "total_files": len(records),
            "target_state_counts": dict(target_counts),
            "action_counts": dict(action_counts),
            "approval_required_paths": [r["path"] for r in records if r["approval_required"]],
        },
        "records": records,
    }


def write_record(
    repo: Path,
    *,
    path: str,
    from_state: str,
    to_state: str,
    decision: str,
    rationale: str,
    agent: str = "local-agent",
) -> dict[str, object]:
    """Append an artifact-promotion decision to the local JSONL ledger."""
    repo = repo.resolve()
    local_dir = repo / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "path": path,
        "from_state": from_state,
        "to_state": to_state,
        "decision": decision,
        "rationale": rationale,
        "agent": agent,
    }
    ledger = local_dir / "artifact_promotion_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def write_manifest_template(repo: Path) -> Path:
    """Write a local artifact-manifest template used during promotion review."""
    repo = repo.resolve()
    local_dir = repo / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    template = local_dir / "ARTIFACT_MANIFEST_TEMPLATE.json"
    if not template.exists():
        template.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "artifact_path": "",
                    "promotion_state": "manifest_backed",
                    "source_script": "",
                    "input_manifest": "",
                    "output_sha256": "",
                    "random_seed": None,
                    "software_version": "",
                    "external_uri": "",
                    "privacy_license_review": "pending",
                    "decision_record": "",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return template


def build_artifact_plan(root: Path, mapping: dict[str, object] | None = None) -> dict[str, object]:
    """Build an artifact promotion plan from a repo map."""
    mapping = mapping or repo_map.build_map(root)
    decisions = [classify_artifact(record) for record in mapping["files"]]
    state_counts = Counter(decision.lifecycle_state for decision in decisions)
    action_counts = Counter(decision.recommended_action for decision in decisions)
    tracked_nonpublic = [
        d.path
        for d in decisions
        if d.tracked and d.lifecycle_state in {"local_scratch", "local_reviewed", "external_artifact"}
    ]
    recommended_next_steps: list[str] = []
    if tracked_nonpublic:
        recommended_next_steps.append(
            "Review tracked local/review/external artifacts and untrack, manifest, or explicitly allow before push."
        )
    if state_counts.get("local_scratch", 0):
        recommended_next_steps.append(
            "Keep local scratch outputs in ignored directories and do not use them as manuscript evidence without promotion."
        )
    if state_counts.get("local_reviewed", 0):
        recommended_next_steps.append(
            "Promote review-required data/outputs through evidence ledgers, result manifests, or public fixtures."
        )
    if state_counts.get("external_artifact", 0):
        recommended_next_steps.append(
            "Move source PDFs, submission artifacts, and large outputs to OSF/Zenodo/private storage; commit only hashes/metadata."
        )

    return {
        "schema_version": "1.0",
        "repo": mapping["repo"],
        "detected_project": mapping["detected_project"],
        "summary": {
            "total_files": len(decisions),
            "lifecycle_state_counts": dict(state_counts),
            "recommended_action_counts": dict(action_counts),
            "tracked_nonpublic_artifacts": tracked_nonpublic,
        },
        "recommended_next_steps": recommended_next_steps,
        "artifacts": [asdict(decision) for decision in decisions],
    }


def build_plan(root: Path, mapping: dict[str, object] | None = None) -> dict[str, object]:
    """Compatibility wrapper used by reorganisation/release helpers."""
    mapping = mapping or repo_map.build_map(root)
    plan = build_artifact_plan(root, mapping)
    promotion = build_promotion_plan(mapping)
    plan["records"] = promotion["records"]
    plan["decisions"] = [
        {
            **record,
            "action": record["recommended_action"],
            "target_state": record["recommended_target_state"],
            "artifact_class": record["current_state"],
        }
        for record in promotion["records"]
    ]
    plan["summary"]["action_counts"] = promotion["summary"]["action_counts"]
    plan["summary"]["target_state_counts"] = promotion["summary"]["target_state_counts"]
    plan["summary"]["approval_required_paths"] = promotion["summary"]["approval_required_paths"]
    return plan


def to_markdown(plan: dict[str, object]) -> str:
    repo = plan["repo"]
    summary = plan["summary"]
    lines = [
        f"# Artifact promotion plan: {repo['name']}",
        "",
        "This plan classifies files into a local-to-public lifecycle. It does not move or publish files.",
        "",
        "## Lifecycle states",
        "",
        "- `local_scratch`: local-only/generated/private; do not commit.",
        "- `local_reviewed`: potentially publishable, but requires human/source review.",
        "- `manifest_backed`: publishable only with schema/provenance/ledger support.",
        "- `external_artifact`: store outside GitHub; commit only metadata/hash when appropriate.",
        "- `public_fixture`: small synthetic or redistributable example/test artifact.",
        "- `public_source`: source, tests, schemas, docs, conductor material.",
        "",
        "## Summary",
        "",
        f"- Total files classified: {summary['total_files']}",
        "",
        "### Lifecycle states",
        "",
    ]
    for key, value in sorted(summary["lifecycle_state_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "### Recommended actions", ""])
    for key, value in sorted(summary["recommended_action_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    tracked_nonpublic = summary["tracked_nonpublic_artifacts"]
    if tracked_nonpublic:
        lines.extend(["", "## Tracked non-public artifacts requiring action", ""])
        for path in tracked_nonpublic[:200]:
            lines.append(f"- `{path}`")
        if len(tracked_nonpublic) > 200:
            lines.append(f"- ... plus {len(tracked_nonpublic) - 200} more")
    lines.extend(["", "## Recommended next steps", ""])
    for step in plan.get("recommended_next_steps", []):
        lines.append(f"- {step}")
    lines.extend(["", "## Artifact inventory", ""])
    lines.append("| Path | State | Action | Destination hint | Tracked | Reasons |")
    lines.append("|---|---|---|---|---:|---|")
    for item in plan["artifacts"][:500]:
        reasons = "; ".join(item["reasons"])
        lines.append(
            f"| `{item['path']}` | `{item['lifecycle_state']}` | `{item['recommended_action']}` | {item['destination_hint']} | {item['tracked']} | {reasons} |"
        )
    if len(plan["artifacts"]) > 500:
        lines.append(f"| ... | ... | ... | ... | ... | {len(plan['artifacts']) - 500} more files omitted |")
    return "\n".join(lines) + "\n"


def load_mapping(root: Path, repo_map_path: Path | None) -> dict[str, object]:
    if repo_map_path:
        return json.loads(repo_map_path.read_text(encoding="utf-8"))
    default = root / ".conductor" / "local" / "repo_map.json"
    if default.exists():
        return json.loads(default.read_text(encoding="utf-8"))
    return repo_map.build_map(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="Repository root")
    parser.add_argument("--repo-map", type=Path, default=None, help="Existing repo_map.json")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()

    root = args.repo.resolve()
    mapping = load_mapping(root, args.repo_map)
    plan = build_artifact_plan(root, mapping)
    default_dir = root / ".conductor" / "local"
    out_json = args.output_json or default_dir / "artifact_promotion_plan.json"
    out_md = args.output_md or default_dir / "artifact_promotion_plan.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(to_markdown(plan), encoding="utf-8")
    if args.write_template:
        template = write_manifest_template(root)
        print(f"Wrote {template}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
