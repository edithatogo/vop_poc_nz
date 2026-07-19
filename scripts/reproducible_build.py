#!/usr/bin/env python3
"""Build twice from one revision and compare byte and archive inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_inventory(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                payload = archive.read(name)
                entries.append(
                    {"path": name, "sha256": _sha256(payload), "size": len(payload)}
                )
        return entries
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to read archive member: {member.name}")
                payload = extracted.read()
                entries.append(
                    {
                        "path": member.name,
                        "sha256": _sha256(payload),
                        "size": len(payload),
                    }
                )
        return entries
    raise ValueError(f"unsupported build artifact: {path.name}")


def artifact_evidence(path: Path) -> dict[str, Any]:
    """Return byte and normalized-content identities for one artifact."""
    payload = path.read_bytes()
    inventory = _archive_inventory(path)
    canonical = json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()
    return {
        "filename": path.name,
        "sha256": _sha256(payload),
        "size": len(payload),
        "inventory_sha256": _sha256(canonical),
        "entries": len(inventory),
    }


def compare_build_directories(first: Path, second: Path) -> dict[str, Any]:
    """Compare two build directories and return a deterministic evidence envelope."""

    def distributions(root: Path) -> dict[str, Path]:
        return {
            path.name: path
            for path in root.iterdir()
            if path.is_file()
            and (path.suffix in {".whl", ".zip"} or path.name.endswith(".tar.gz"))
        }

    first_paths = distributions(first)
    second_paths = distributions(second)
    names_match = first_paths.keys() == second_paths.keys()
    artifact_set_complete = (
        len(first_paths) == 2
        and sum(name.endswith(".whl") for name in first_paths) == 1
        and sum(name.endswith(".tar.gz") for name in first_paths) == 1
    )
    artifacts: list[dict[str, Any]] = []
    for name in sorted(first_paths.keys() & second_paths.keys()):
        left = artifact_evidence(first_paths[name])
        right = artifact_evidence(second_paths[name])
        artifacts.append(
            {
                "filename": name,
                "byte_identical": left["sha256"] == right["sha256"],
                "inventory_identical": left["inventory_sha256"]
                == right["inventory_sha256"],
                "first": left,
                "second": right,
            }
        )
    return {
        "schema_version": "1.0.0",
        "names_match": names_match,
        "artifact_set_complete": artifact_set_complete,
        "reproducible": names_match
        and artifact_set_complete
        and bool(artifacts)
        and all(item["byte_identical"] for item in artifacts),
        "artifacts": artifacts,
    }


def _source_date_epoch(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%ct"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_reproducible_build(
    repo: Path, *, output_dir: Path | None = None
) -> dict[str, Any]:
    """Run two isolated uv builds from the same source revision."""
    environment = {**os.environ, "SOURCE_DATE_EPOCH": _source_date_epoch(repo)}
    with (
        tempfile.TemporaryDirectory(prefix="vop-build-a-") as first_temp,
        tempfile.TemporaryDirectory(prefix="vop-build-b-") as second_temp,
    ):
        first = Path(first_temp)
        second = Path(second_temp)
        for destination in (first, second):
            subprocess.run(
                ["uv", "build", "--out-dir", str(destination)],
                cwd=repo,
                env=environment,
                check=True,
            )
        report = compare_build_directories(first, second)
        if output_dir is not None and report["reproducible"]:
            output_dir.mkdir(parents=True, exist_ok=True)
            for artifact in first.iterdir():
                if artifact.is_file():
                    shutil.copy2(artifact, output_dir / artifact.name)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path)
    args = parser.parse_args()
    report = verify_reproducible_build(args.repo.resolve(), output_dir=args.dist_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
