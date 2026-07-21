#!/usr/bin/env python3
"""Fail-closed validation for the arXiv paper-update evidence contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(manifest: dict, root: Path) -> None:
    required = {"schema_version", "source_revision", "package_version", "agents", "claims", "artifacts", "submission_ready"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest["schema_version"] != "1.0.0" or not isinstance(manifest["source_revision"], str) or len(manifest["source_revision"]) != 40:
        raise ValueError("invalid manifest identity")
    if len(manifest["agents"]) < 4:
        raise ValueError("at least four agent receipts are required")
    agent_ids = {entry.get("id") for entry in manifest["agents"]}
    if {"literature", "methods", "reproducibility", "editorial", "integrator"} - agent_ids:
        raise ValueError("required agent receipt missing")
    if not manifest["claims"] or any(not entry.get("evidence") for entry in manifest["claims"]):
        raise ValueError("every claim requires evidence")
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {artifact.get('path')}")
    tool_ids = {entry.get("id") for entry in manifest["agents"]}
    if "sourceright" not in tool_ids or "authentext" not in tool_ids:
        raise ValueError("Sourceright and AuthenText receipts are required")
    if manifest["submission_ready"] is not False:
        raise ValueError("submission_ready must remain false until human author approval")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    validate(json.loads(args.manifest.read_text(encoding="utf-8")), args.root.resolve())
    print("paper-update-contract: valid and human-gated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
