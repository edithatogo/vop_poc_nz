"""Cross-platform normalized archive identity for C15 assurance."""

from __future__ import annotations

import json
import tarfile
import zipfile
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TypedDict

_TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".csv",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_NORMALIZATION = "sorted-paths+utf8-lf+content-sha256+derived-record-excluded"


class ArtifactMismatch(ValueError):
    """Raised when normalized artifacts do not have identical content."""


class ArchiveEntry(TypedDict):
    """Normalized content identity for one archive member."""

    path: str
    sha256: str
    size: int


def _safe_path(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path.as_posix()


def _normalized_content(name: str, content: bytes) -> bytes:
    if PurePosixPath(name).suffix.casefold() not in _TEXT_SUFFIXES:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _zip_entries(path: Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.is_dir():
                yield member.filename, archive.read(member)


def _tar_entries(path: Path) -> Iterable[tuple[str, bytes]]:
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member cannot be read: {member.name}")
            yield member.name, extracted.read()


def _entries(path: Path) -> list[ArchiveEntry]:
    if zipfile.is_zipfile(path):
        raw_entries = _zip_entries(path)
    elif tarfile.is_tarfile(path):
        raw_entries = _tar_entries(path)
    else:
        raise ValueError(f"unsupported archive format: {path.name}")
    entries: list[ArchiveEntry] = []
    seen: set[str] = set()
    for raw_name, raw_content in raw_entries:
        name = _safe_path(raw_name)
        if name.endswith(".dist-info/RECORD"):
            # RECORD hashes raw wheel bytes, including platform checkout line endings.
            # Every underlying member is independently normalized and hashed below.
            continue
        if name in seen:
            raise ValueError(f"duplicate archive path: {name}")
        seen.add(name)
        content = _normalized_content(name, raw_content)
        entries.append(
            {"path": name, "sha256": sha256(content).hexdigest(), "size": len(content)}
        )
    if not entries:
        raise ValueError("archive contains no regular files")
    return sorted(entries, key=lambda item: item["path"])


def normalized_archive_report(path: Path, *, runner: str) -> dict[str, object]:
    """Return metadata-independent content identity for a zip, wheel, or tar archive."""
    if not runner.strip():
        raise ValueError("runner identity must not be empty")
    entries = _entries(path)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "1.0.0",
        "artifact_name": path.name,
        "runner": runner,
        "normalization": _NORMALIZATION,
        "normalized_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
        "entries": entries,
    }


def compare_digest_reports(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    """Fail closed unless two independently generated reports have equal identity."""
    for label, report in (("left", left), ("right", right)):
        if report.get("schema_version") != "1.0.0":
            raise ArtifactMismatch(f"{label} digest report schema is unsupported")
        if report.get("normalization") != _NORMALIZATION:
            raise ArtifactMismatch(f"{label} normalization policy is unsupported")
        digest = report.get("normalized_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ArtifactMismatch(f"{label} normalized digest is invalid")
        if not isinstance(report.get("entries"), list):
            raise ArtifactMismatch(f"{label} archive inventory is invalid")
    if left.get("runner") == right.get("runner"):
        raise ArtifactMismatch("independent runner identities must differ")
    if left["normalized_sha256"] != right["normalized_sha256"]:
        raise ArtifactMismatch("normalized artifact digests differ")
    if left["entries"] != right["entries"]:
        raise ArtifactMismatch("normalized archive inventories differ")
    return {
        "schema_version": "1.0.0",
        "matched": True,
        "normalized_sha256": left["normalized_sha256"],
        "runners": [left.get("runner"), right.get("runner")],
    }


__all__ = [
    "ArtifactMismatch",
    "compare_digest_reports",
    "normalized_archive_report",
]
