"""Cross-platform normalized archive identity for C15 assurance."""

from __future__ import annotations

import base64
import csv
import io
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
        ".template",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_TEXT_FILENAMES = frozenset(
    {
        "authors",
        "changelog",
        "copying",
        "license",
        "notice",
        "readme",
        "snakefile",
    }
)
_NORMALIZATION = "sorted-paths+declared-utf8-text-lf+content-sha256+record-semantics-v1"


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
    path = PurePosixPath(name)
    filename = path.name.casefold()
    stem = filename.split(".", maxsplit=1)[0]
    if path.suffix.casefold() not in _TEXT_SUFFIXES and stem not in _TEXT_FILENAMES:
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


def _wheel_record_entry(
    raw: dict[str, bytes], record_name: str, normalized: list[ArchiveEntry]
) -> ArchiveEntry:
    """Validate raw wheel RECORD integrity and bind its normalized semantics."""
    try:
        rows = list(csv.reader(io.StringIO(raw[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("wheel RECORD is not valid UTF-8 CSV") from exc
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise ValueError("wheel RECORD rows must contain path, digest, and size")
        name = _safe_path(row[0])
        if name in recorded:
            raise ValueError(f"duplicate wheel RECORD path: {name}")
        recorded.add(name)
        digest, size = row[1], row[2]
        if name == record_name:
            if digest or size:
                raise ValueError("wheel RECORD must not hash itself")
            continue
        content = raw.get(name)
        if content is None:
            raise ValueError(f"wheel RECORD references missing member: {name}")
        expected_digest = (
            base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=").decode()
        )
        if digest != f"sha256={expected_digest}" or size != str(len(content)):
            raise ValueError(f"wheel RECORD integrity mismatch: {name}")
    if recorded != set(raw):
        raise ValueError("wheel RECORD inventory does not match archive inventory")

    normalized_rows = [
        [entry["path"], f"sha256={entry['sha256']}", str(entry["size"])]
        for entry in normalized
    ]
    normalized_rows.append([record_name, "", ""])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(sorted(normalized_rows))
    content = stream.getvalue().encode("utf-8")
    return {
        "path": record_name,
        "sha256": sha256(content).hexdigest(),
        "size": len(content),
    }


def _raw_archive(path: Path) -> dict[str, bytes]:
    if zipfile.is_zipfile(path):
        raw_entries = _zip_entries(path)
    elif tarfile.is_tarfile(path):
        raw_entries = _tar_entries(path)
    else:
        raise ValueError(f"unsupported archive format: {path.name}")
    raw: dict[str, bytes] = {}
    for raw_name, raw_content in raw_entries:
        name = _safe_path(raw_name)
        if name in raw:
            raise ValueError(f"duplicate archive path: {name}")
        raw[name] = raw_content
    return raw


def _entries(path: Path) -> list[ArchiveEntry]:
    raw = _raw_archive(path)
    record_names = [name for name in raw if name.endswith(".dist-info/RECORD")]
    if path.suffix == ".whl" and len(record_names) != 1:
        raise ValueError("wheel must contain exactly one .dist-info/RECORD")
    if len(record_names) > 1:
        raise ValueError("archive contains multiple .dist-info/RECORD files")

    entries: list[ArchiveEntry] = []
    for name, raw_content in raw.items():
        if name in record_names:
            continue
        content = _normalized_content(name, raw_content)
        entries.append(
            {"path": name, "sha256": sha256(content).hexdigest(), "size": len(content)}
        )
    entries.sort(key=lambda item: item["path"])
    if record_names:
        entries.append(_wheel_record_entry(raw, record_names[0], entries))
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
