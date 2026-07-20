#!/usr/bin/env python3
"""Check internal consistency of the duplicated v6 implementation-pack surfaces.

The live repositories must be merged architecture-aware, but the implementation
pack itself should be deterministic. This check detects version drift, stale
reference copies in overlays, reintroduced legacy registries, generated-issue
drift, and cache artifacts before packaging.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import issue_registry

PROJECTS = ("vop_poc_nz", "voiage")


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _compare_tree(source: Path, target: Path, label: str, findings: list[Finding]) -> None:
    source_files = _relative_files(source)
    target_files = _relative_files(target)
    for rel in sorted(source_files - target_files):
        findings.append(Finding("error", "overlay_missing", f"{label} missing {rel}"))
    for rel in sorted(source_files & target_files):
        if (source / rel).read_bytes() != (target / rel).read_bytes():
            findings.append(Finding("error", "overlay_drift", f"{label} differs at {rel}"))


def _extract_contract_version(path: Path) -> str | None:
    match = re.search(r'^METHOD_CONTRACT_VERSION\s*=\s*["\']([^"\']+)["\']', path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def validate(pack_root: Path) -> dict[str, object]:
    pack_root = pack_root.resolve()
    findings: list[Finding] = []
    version = (pack_root / "PACK_VERSION").read_text(encoding="utf-8").strip()
    pack = _json(pack_root / "conductor" / "pack.json")
    manifest = _json(pack_root / "conductor" / "manifest.json")
    integration = _json(pack_root / "integration" / "manifest.json")
    versions = {
        "PACK_VERSION": version,
        "conductor/pack.json": str(pack.get("version")),
        "conductor/manifest.json": str(manifest.get("pack_version")),
        "integration/manifest.json": str(integration.get("pack_version")),
    }
    if len(set(versions.values())) != 1:
        findings.append(Finding("error", "pack_version_drift", f"pack versions disagree: {versions}"))

    contract = str(pack.get("method_contract_version"))
    fixture = _json(pack_root / "fixtures" / "perspective" / "conformance_v1.json")
    contract_versions = {
        "pack": contract,
        "fixture": str(fixture.get("method_contract_version")),
        "vop_poc_nz": str(_extract_contract_version(pack_root / "overlays" / "vop_poc_nz" / "src" / "vop_poc_nz" / "perspective.py")),
        "voiage": str(_extract_contract_version(pack_root / "overlays" / "voiage" / "src" / "voiage" / "perspective" / "core.py")),
    }
    if len(set(contract_versions.values())) != 1:
        findings.append(Finding("error", "method_contract_drift", f"method contract versions disagree: {contract_versions}"))

    for project in PROJECTS:
        overlay = pack_root / "overlays" / project
        _compare_tree(pack_root / "conductor", overlay / "conductor", f"{project}:conductor", findings)
        _compare_tree(pack_root / "integration", overlay / "integration", f"{project}:integration", findings)
        _compare_tree(pack_root / "issues", overlay / "issues", f"{project}:issues", findings)
        _compare_tree(pack_root / "local_agent" / "prompts", overlay / "prompts" / "local_agent", f"{project}:prompts", findings)
        _compare_tree(pack_root / "method", overlay / "docs" / "method", f"{project}:method", findings)
        for source in sorted((pack_root / "scripts").glob("*.py")):
            target = overlay / "scripts" / source.name
            if not target.exists():
                findings.append(Finding("error", "overlay_missing", f"{project}:scripts missing {source.name}"))
            elif source.read_bytes() != target.read_bytes():
                findings.append(Finding("error", "overlay_drift", f"{project}:scripts differs at {source.name}"))
        for source in sorted((pack_root / "schemas").glob("*.json")):
            target = overlay / "schemas" / source.name
            if not target.exists():
                findings.append(Finding("error", "overlay_missing", f"{project}:schemas missing {source.name}"))
            elif source.read_bytes() != target.read_bytes():
                findings.append(Finding("error", "overlay_drift", f"{project}:schemas differs at {source.name}"))
        if (pack_root / "AGENTS.md").read_bytes() != (overlay / "AGENTS.md").read_bytes():
            findings.append(Finding("error", "overlay_drift", f"{project}:AGENTS.md differs from the pack reference"))

    legacy_tracks = sorted((pack_root / "conductor" / "tracks").glob("track_*.md"))
    if legacy_tracks:
        findings.append(Finding("error", "legacy_track_active", f"legacy active tracks remain: {[path.name for path in legacy_tracks]}"))
    old_prompts = [path for path in (pack_root / "local_agent" / "prompts").glob("*.md") if not re.fullmatch(r"P\d\d_[A-Z0-9_]+\.md", path.name)]
    if old_prompts:
        findings.append(Finding("error", "legacy_prompt_active", f"non-canonical prompts remain: {[path.name for path in old_prompts]}"))

    issue_drift = issue_registry.check(pack_root)
    for item in issue_drift:
        findings.append(Finding("error", "generated_issue_drift", item))

    try:
        tracked_output = subprocess.run(
            ["git", "ls-files"],
            cwd=pack_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        tracked_output = []
    tracked_caches = [path for path in tracked_output if "__pycache__/" in path or path.endswith(".pyc")]
    if tracked_caches:
        findings.append(Finding("error", "cache_artifact", f"tracked cache artifacts present: {tracked_caches[:20]}"))

    summary = {
        "valid": not any(item.severity == "error" for item in findings),
        "errors": sum(item.severity == "error" for item in findings),
        "warnings": sum(item.severity == "warning" for item in findings),
        "pack_version": version,
        "method_contract_version": contract,
    }
    return {
        "schema_version": "1.0",
        "pack_root": str(pack_root),
        "summary": summary,
        "versions": versions,
        "contract_versions": contract_versions,
        "findings": [asdict(item) for item in findings],
    }


def to_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# Pack self-check",
        "",
        f"- Valid: **{summary['valid']}**",
        f"- Errors: **{summary['errors']}**",
        f"- Warnings: **{summary['warnings']}**",
        f"- Pack version: `{summary['pack_version']}`",
        f"- Method contract: `{summary['method_contract_version']}`",
        "",
        "## Findings",
        "",
    ]
    findings = report["findings"]
    lines.extend(f"- **{item['severity'].upper()} `{item['code']}`:** {item['message']}" for item in findings) if findings else lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    report = validate(args.pack_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report), end="")
    return 0 if report["summary"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
