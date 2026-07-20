#!/usr/bin/env python3
"""Inspect a live repository and produce a safe v6 integration plan.

The doctor never modifies source files. It detects package layout, existing
perspective/conductor/agent surfaces, maps overlay paths to the live tree, and
classifies each proposed file as safe-add, identical, or merge-required.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_project(repo: Path) -> dict[str, Any]:
    pyproject = repo / "pyproject.toml"
    name = repo.name
    package_layout = "unknown"
    package_root = None
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            name = str(data.get("project", {}).get("name", name)).replace("-", "_")
            find = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
            where = find.get("where")
            if isinstance(where, list) and where:
                package_layout = str(where[0])
        except (tomllib.TOMLDecodeError, OSError):
            pass
    candidates = [repo / "src" / name, repo / name]
    for candidate in candidates:
        if candidate.exists():
            package_root = candidate
            package_layout = "src" if candidate.parent.name == "src" else "flat"
            break
    normalised_repo = repo.name.replace("-", "_")
    kind = "voiage" if name == "voiage" or normalised_repo == "voiage" else "vop_poc_nz" if name == "vop_poc_nz" or normalised_repo == "vop_poc_nz" else "unknown"
    return {"kind": kind, "project_name": name, "package_layout": package_layout, "package_root": str(package_root) if package_root else None}


def load_integration_manifest(pack_root: Path) -> dict[str, Any]:
    return json.loads((pack_root / "integration" / "manifest.json").read_text(encoding="utf-8"))


def rule_mode(manifest: dict[str, Any], repository: str, overlay_rel: str) -> tuple[str, str]:
    mode = str(manifest.get("default_mode", "add_if_absent"))
    reason = "default add-if-absent policy"
    for rule in manifest.get("rules", []):
        repo_rule = str(rule.get("repository", "*"))
        if repo_rule not in {"*", repository}:
            continue
        if fnmatch.fnmatch(overlay_rel, str(rule.get("pattern", ""))):
            mode = str(rule.get("mode", mode))
            reason = str(rule.get("reason", reason))
    return mode, reason


def map_target(repo: Path, project: dict[str, Any], overlay_rel: str) -> Path:
    if project["kind"] == "voiage" and overlay_rel.startswith("src/voiage/") and project["package_layout"] == "flat":
        return repo / overlay_rel.removeprefix("src/")
    return repo / overlay_rel


def existing_method_signals(repo: Path) -> dict[str, list[str]]:
    perspective = []
    for path in repo.rglob("*.py"):
        rel_path = path.relative_to(repo)
        if any(part in {".git", ".venv", "venv", "site-packages", "__pycache__", ".conductor"} for part in rel_path.parts):
            continue
        rel = str(rel_path)
        lower = rel.casefold()
        if "perspective" in lower or "value_of_information" in lower or "frontier" in lower:
            perspective.append(rel)
    agents = [str(path.relative_to(repo)) for path in repo.iterdir() if path.is_file() and path.name.casefold() == "agents.md"]
    conductor = [str(path.relative_to(repo)) for path in (repo / "conductor").rglob("*") if path.is_file()] if (repo / "conductor").exists() else []
    return {"perspective_files": sorted(perspective)[:100], "agent_files": sorted(agents), "conductor_files": sorted(conductor)[:100]}


def build_report(repo: Path, pack_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    pack_root = pack_root.resolve()
    project = detect_project(repo)
    if project["kind"] == "unknown":
        raise ValueError(f"Could not identify target repository at {repo}")
    overlay_root = pack_root / "overlays" / project["kind"]
    if not overlay_root.exists():
        raise ValueError(f"Missing reference overlay for {project['kind']}: {overlay_root}")
    manifest = load_integration_manifest(pack_root)
    signals = existing_method_signals(repo)
    items = []
    for source in sorted(
        path for path in overlay_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ):
        rel = str(source.relative_to(overlay_root))
        target = map_target(repo, project, rel)
        mode, reason = rule_mode(manifest, project["kind"], rel)
        status = "safe_add"
        if target.exists():
            status = "same" if sha256(source) == sha256(target) else "merge_required"
        elif mode in {"merge_required", "managed_block"}:
            status = mode
        if project["kind"] == "voiage" and rel.startswith("src/voiage/perspective/") and signals["perspective_files"]:
            status = "merge_required"
            reason = "existing perspective/frontier implementation detected"
        items.append({"source": str(source), "overlay_path": rel, "target": str(target), "mode": mode, "status": status, "reason": reason})
    summary = {key: sum(item["status"] == key for item in items) for key in {"safe_add", "same", "merge_required", "managed_block"}}
    warnings = []
    if any(name == "agents.md" for name in signals["agent_files"]):
        warnings.append("Lower-case agents.md detected; merge into canonical AGENTS.md on case-sensitive filesystems.")
    if signals["conductor_files"]:
        warnings.append("Existing conductor directory detected; use registry migration rather than replacing it.")
    if project["kind"] == "voiage" and signals["perspective_files"]:
        warnings.append("Existing voiage perspective/frontier files detected; do not apply the reference perspective overlay wholesale.")
    return {"schema_version": "1.0", "pack_version": "6.0.0", "repo": str(repo), "project": project, "signals": signals, "summary": summary, "warnings": warnings, "items": items}


def to_markdown(report: dict[str, Any]) -> str:
    p = report["project"]
    lines = [
        f"# Pack doctor: {p['project_name']}", "",
        f"- Repository kind: `{p['kind']}`",
        f"- Package layout: `{p['package_layout']}`",
        f"- Safe additions: **{report['summary'].get('safe_add', 0)}**",
        f"- Existing identical files: **{report['summary'].get('same', 0)}**",
        f"- Merge-required files: **{report['summary'].get('merge_required', 0)}**",
        f"- Managed agent blocks: **{report['summary'].get('managed_block', 0)}**",
        "", "## Warnings", "",
    ]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["- None."]
    lines += ["", "## Integration items", "", "| Status | Overlay path | Target | Reason |", "|---|---|---|---|"]
    for item in report["items"]:
        target = Path(item["target"]).name if len(item["target"]) > 80 else item["target"]
        lines.append(f"| {item['status']} | `{item['overlay_path']}` | `{target}` | {item['reason']} |")
    lines += ["", "## Rule", "", "Only `safe_add` items may be copied automatically. Review and merge every other item in the context of the live repository.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = build_report(args.repo, args.pack_root)
    out = args.repo / ".conductor" / "local"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack_doctor.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "pack_doctor.md").write_text(to_markdown(report), encoding="utf-8")
    print(out / "pack_doctor.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
