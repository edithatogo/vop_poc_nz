#!/usr/bin/env python3
"""Audit frozen, supported-latest, absolute-stable, and prerelease dependency lanes."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.request
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

Lane = Literal["stable", "frontier"]


def declared_requirements(
    config: Mapping[str, object],
) -> list[tuple[str, Requirement]]:
    """Collect core, optional, development, and build declarations."""
    project = config.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject requires a project table")
    grouped: list[tuple[str, object]] = [("core", project.get("dependencies", []))]
    optional = project.get("optional-dependencies", {})
    groups = config.get("dependency-groups", {})
    build = config.get("build-system", {})
    if not isinstance(optional, dict) or not isinstance(groups, dict):
        raise ValueError("dependency declarations must be tables")
    grouped.extend((f"optional:{name}", values) for name, values in optional.items())
    grouped.extend((f"group:{name}", values) for name, values in groups.items())
    grouped.append(
        ("build", build.get("requires", []) if isinstance(build, dict) else [])
    )
    return [
        (scope, Requirement(item))
        for scope, values in grouped
        if isinstance(values, list)
        for item in values
        if isinstance(item, str)
    ]


def locked_versions(lock: Mapping[str, object]) -> dict[str, Version]:
    """Read exact selected versions from uv.lock's package inventory."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock requires a package array")
    result: dict[str, Version] = {}
    for package in packages:
        if not isinstance(package, dict):
            continue
        name, version = package.get("name"), package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            result[canonicalize_name(name)] = Version(version)
    return result


def fetch_pypi_project(name: str) -> dict[str, Any]:
    """Read official release metadata; this is used only by the ephemeral lane."""
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid PyPI response for {name}")
    return payload


def _supports_python(files: object, python_version: Version) -> bool:
    if not isinstance(files, list) or not files:
        return False
    for file in files:
        if not isinstance(file, dict) or file.get("yanked") is True:
            continue
        requires_python = file.get("requires_python")
        if requires_python is None:
            return True
        if isinstance(requires_python, str) and SpecifierSet(requires_python).contains(
            python_version, prereleases=True
        ):
            return True
    return False


def release_frontier(
    payload: Mapping[str, object],
    requirement: Requirement,
    *,
    python_version: Version,
) -> dict[str, str | None]:
    """Classify supported stable, absolute stable, and prerelease releases."""
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        raise ValueError(f"PyPI releases missing for {requirement.name}")
    stable: list[Version] = []
    prerelease: list[Version] = []
    for raw, files in releases.items():
        if not isinstance(raw, str):
            continue
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if not _supports_python(files, python_version):
            continue
        (prerelease if version.is_prerelease else stable).append(version)
    absolute = max(stable, default=None)
    preview = max(prerelease, default=None)
    supported = max(
        (version for version in stable if requirement.specifier.contains(version)),
        default=None,
    )
    return {
        "supported_latest_stable": str(supported) if supported else None,
        "absolute_latest_stable": str(absolute) if absolute else None,
        "latest_prerelease": str(preview) if preview else None,
    }


def dependency_report(
    requirements: Sequence[tuple[str, Requirement]],
    locked: Mapping[str, Version],
    *,
    lane: Lane,
    python_version: Version,
    pypi: Mapping[str, Mapping[str, object]] | None = None,
    lock_digest: str,
) -> dict[str, object]:
    """Build truthful lane evidence without treating prereleases as stable debt."""
    rows: list[dict[str, object]] = []
    for scope, requirement in requirements:
        key = canonicalize_name(requirement.name)
        selected = locked.get(key)
        lock_managed = scope != "build"
        frozen_valid = (
            selected is not None
            and requirement.specifier.contains(selected, prereleases=True)
            if lock_managed
            else None
        )
        row: dict[str, object] = {
            "scope": scope,
            "package": requirement.name,
            "specifier": str(requirement.specifier),
            "locked": str(selected) if selected else None,
            "stable_frozen_valid": frozen_valid,
            "resolution_policy": (
                "uv_lock" if lock_managed else "build_frontend_isolated"
            ),
        }
        if lane == "frontier":
            if pypi is None or key not in pypi:
                raise ValueError(f"missing PyPI evidence for {requirement.name}")
            frontier: dict[str, object] = dict(
                release_frontier(pypi[key], requirement, python_version=python_version)
            )
            supported = frontier["supported_latest_stable"]
            at_supported = (
                selected is not None and str(selected) == supported
                if lock_managed
                else None
            )
            frontier.update(
                at_supported_latest=at_supported,
                absolute_stable_blocked_by_policy=(
                    frontier["absolute_latest_stable"] is not None
                    and frontier["absolute_latest_stable"] != supported
                ),
                prerelease_policy="report_only",
            )
            row.update(frontier)
        rows.append(row)
    frozen_passed = all(row["stable_frozen_valid"] is not False for row in rows)
    frontier_passed = lane == "stable" or all(
        row["at_supported_latest"] is not False for row in rows
    )
    return {
        "schema_version": "2.0.0",
        "lane": lane,
        "python_version": str(python_version),
        "lock_sha256": lock_digest,
        "stable_frozen_passed": frozen_passed,
        "supported_latest_passed": frontier_passed,
        "absolute_and_prerelease_are_observational": True,
        "passed": frozen_passed and frontier_passed,
        "dependencies": rows,
    }


def _markdown(report: Mapping[str, object]) -> str:
    rows = report["dependencies"]
    assert isinstance(rows, list)
    lines = [
        "# Dependency policy evidence",
        "",
        f"Lane: `{report['lane']}`",
        f"Frozen lock valid: **{report['stable_frozen_passed']}**",
        f"Supported-latest resolved: **{report['supported_latest_passed']}**",
        "",
        "| Scope | Package | Locked | Supported latest stable | Absolute stable | Prerelease |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        assert isinstance(row, dict)
        typed_row = cast(dict[str, object], row)
        lines.append(
            f"| `{typed_row['scope']}` | `{typed_row['package']}` | "
            f"`{typed_row['locked']}` | "
            f"`{typed_row.get('supported_latest_stable')}` | "
            f"`{typed_row.get('absolute_latest_stable')}` | "
            f"`{typed_row.get('latest_prerelease')}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--lane", choices=("stable", "frontier"), default="stable")
    parser.add_argument(
        "--python-version",
        default=f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    lock_bytes = (repo / "uv.lock").read_bytes()
    locked = locked_versions(tomllib.loads(lock_bytes.decode("utf-8")))
    requirements = declared_requirements(pyproject)
    pypi: dict[str, Mapping[str, object]] | None = None
    if args.lane == "frontier":
        pypi = {
            key: fetch_pypi_project(requirement.name)
            for key, requirement in {
                canonicalize_name(item.name): item for _, item in requirements
            }.items()
        }
    report = dependency_report(
        requirements,
        locked,
        lane=args.lane,
        python_version=Version(args.python_version),
        pypi=pypi,
        lock_digest=sha256(lock_bytes).hexdigest(),
    )
    output = args.output or repo / ".conductor/local/dependency_frontier.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(output)
    return 2 if args.strict and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
