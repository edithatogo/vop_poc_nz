#!/usr/bin/env python3
"""Audit version, licence, Python support, citation, and README metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cff_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*['\"]?([^'\"\n]+)", text)
    return match.group(1).strip() if match else None


def _license_from_text(text: str) -> str | None:
    lower = text.casefold()
    if "apache license" in lower and "version 2.0" in lower:
        return "Apache-2.0"
    if "mit license" in lower or "permission is hereby granted, free of charge" in lower:
        return "MIT"
    if "bsd 3-clause" in lower:
        return "BSD-3-Clause"
    return None


def build_audit(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    findings: list[Finding] = []
    values: dict[str, Any] = {}
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            values["project_name"] = project.get("name")
            values["pyproject_version"] = project.get("version")
            values["requires_python"] = project.get("requires-python")
            values["classifiers"] = project.get("classifiers", [])
            license_value = project.get("license")
            if isinstance(license_value, str):
                values["pyproject_license"] = license_value
            elif isinstance(license_value, dict):
                values["pyproject_license"] = license_value.get("text") or license_value.get("file")
        except tomllib.TOMLDecodeError as exc:
            findings.append(Finding("error", "invalid_pyproject", str(exc)))
    else:
        findings.append(Finding("error", "missing_pyproject", "pyproject.toml is missing"))

    citation = repo / "CITATION.cff"
    if citation.exists():
        text = citation.read_text(encoding="utf-8", errors="replace")
        values["citation_version"] = _cff_scalar(text, "version")
        values["citation_title"] = _cff_scalar(text, "title")
        values["citation_license"] = _cff_scalar(text, "license")
    else:
        findings.append(Finding("warning", "missing_citation", "CITATION.cff is missing"))

    license_path = repo / "LICENSE"
    if license_path.exists():
        values["detected_license"] = _license_from_text(license_path.read_text(encoding="utf-8", errors="replace")[:12000])
    else:
        findings.append(Finding("error", "missing_license", "LICENSE is missing"))

    package_version = None
    for init_path in list((repo / "src").glob("*/__init__.py")) + list(repo.glob("*/__init__.py")):
        match = re.search(r"(?m)^__version__\s*=\s*['\"]([^'\"]+)", init_path.read_text(encoding="utf-8", errors="replace"))
        if match:
            package_version = match.group(1)
            values["package_version_file"] = str(init_path.relative_to(repo))
            break
    values["package_version"] = package_version
    tag = _git(repo, "describe", "--tags", "--abbrev=0")
    values["latest_tag"] = tag

    declared_versions = {
        str(value).removeprefix("v")
        for key, value in values.items()
        if key in {"pyproject_version", "citation_version", "package_version"}
        and value
    }
    if len(declared_versions) > 1:
        findings.append(
            Finding(
                "error",
                "version_mismatch",
                f"Declared version values disagree: {sorted(declared_versions)}",
            )
        )
    latest_tag = str(values.get("latest_tag") or "").removeprefix("v")
    if latest_tag and declared_versions and latest_tag not in declared_versions:
        findings.append(
            Finding(
                "warning",
                "unreleased_version",
                f"Declared version {sorted(declared_versions)[0]} is newer than latest tag {latest_tag}; do not describe it as released.",
            )
        )

    detected_license = values.get("detected_license")
    citation_license = values.get("citation_license")
    if detected_license and citation_license and str(citation_license) != detected_license:
        findings.append(Finding("error", "license_mismatch", f"LICENSE is {detected_license}, CITATION.cff says {citation_license}"))
    project_license = str(values.get("pyproject_license") or "")
    if detected_license and project_license and detected_license.casefold() not in project_license.casefold() and project_license.casefold() not in {"license", "license file"}:
        if project_license not in {"LICENSE", "LICENSE.txt"}:
            findings.append(Finding("warning", "pyproject_license_ambiguous", f"pyproject licence metadata ({project_license}) does not clearly match {detected_license}"))

    readme = repo / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="replace")
        placeholders = [token for token in ("[Your Name]", "[repository-url]", "[funding sources]", "your.email@example.com") if token.casefold() in text.casefold()]
        if placeholders:
            findings.append(Finding("error", "readme_placeholders", f"README contains placeholders: {', '.join(placeholders)}"))
        if "from src." in text or "from src import" in text:
            findings.append(Finding("error", "readme_src_import", "README imports from src rather than the installed package"))
        if detected_license == "Apache-2.0" and re.search(r"(?i)licensed under the MIT license|license:\s*MIT", text):
            findings.append(Finding("error", "readme_license_mismatch", "README says MIT while LICENSE is Apache-2.0"))
        if detected_license == "MIT" and "Apache 2.0" in text:
            findings.append(Finding("error", "readme_license_mismatch", "README says Apache-2.0 while LICENSE is MIT"))
    else:
        findings.append(Finding("error", "missing_readme", "README.md is missing"))

    agent_files = [path.name for path in repo.iterdir() if path.is_file() and path.name.casefold() == "agents.md"]
    values["agent_files"] = agent_files
    if "agents.md" in agent_files and "AGENTS.md" not in agent_files:
        findings.append(Finding("warning", "lowercase_agents", "Only lower-case agents.md is present; consolidate into AGENTS.md"))
    if len(agent_files) > 1:
        findings.append(Finding("error", "duplicate_agents", f"Multiple case-variant agent files exist: {agent_files}"))

    return {
        "schema_version": "1.0",
        "repo": str(repo),
        "summary": {
            "errors": sum(item.severity == "error" for item in findings),
            "warnings": sum(item.severity == "warning" for item in findings),
            "consistent": not any(item.severity == "error" for item in findings),
        },
        "values": values,
        "findings": [asdict(item) for item in findings],
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Metadata consistency audit", "", f"Consistent: **{report['summary']['consistent']}** · Errors: **{report['summary']['errors']}** · Warnings: **{report['summary']['warnings']}**", "", "## Values", ""]
    for key, value in sorted(report["values"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Findings", ""]
    lines += [f"- **{item['severity'].upper()} `{item['code']}`:** {item['message']}" for item in report["findings"]] or ["- None."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = build_audit(args.repo)
    out = args.repo / ".conductor" / "local"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_consistency.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "metadata_consistency.md").write_text(to_markdown(report), encoding="utf-8")
    print(out / "metadata_consistency.md")
    return 2 if args.strict and not report["summary"]["consistent"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
