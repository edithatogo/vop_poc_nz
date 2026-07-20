"""Deterministic, cross-platform manifests for publishable evidence files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
)
MANIFEST_VERSION = "1.0.0"


def _canonical_bytes(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return raw, "binary"
    text = raw.decode("utf-8-sig")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return canonical, "utf8-lf"


def evidence_entry(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Describe one file using a repository-relative path and canonical digest."""
    repo = Path(root).resolve()
    source = Path(path).resolve()
    relative = source.relative_to(repo).as_posix()
    canonical, normalization = _canonical_bytes(source)
    return {
        "path": relative,
        "sha256": sha256(canonical).hexdigest(),
        "bytes": len(canonical),
        "normalization": normalization,
    }


def build_evidence_manifest(
    paths: Iterable[str | Path], *, root: str | Path
) -> dict[str, Any]:
    """Build a deterministic manifest independent of OS paths and line endings."""
    entries = sorted(
        (evidence_entry(path, root=root) for path in paths),
        key=lambda item: item["path"],
    )
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Evidence manifest paths must be unique")
    return {
        "manifest_version": MANIFEST_VERSION,
        "algorithm": "sha256",
        "files": entries,
    }


def write_evidence_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write canonical JSON with stable ordering and LF line endings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def verify_evidence_manifest(
    manifest: Mapping[str, Any], *, root: str | Path
) -> list[str]:
    """Return verification failures; an empty list means the manifest is current."""
    repo = Path(root).resolve()
    failures: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        failures.append(
            f"unsupported manifest_version: {manifest.get('manifest_version')!r}"
        )
    for expected in manifest.get("files", []):
        relative = str(expected["path"])
        source = (repo / relative).resolve()
        try:
            source.relative_to(repo)
        except ValueError:
            failures.append(f"path escapes root: {relative}")
            continue
        if not source.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = evidence_entry(source, root=repo)
        for key in ("sha256", "bytes", "normalization"):
            if actual[key] != expected.get(key):
                failures.append(f"{relative}: {key} mismatch")
    return failures
