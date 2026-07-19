"""Cross-platform normalized archive identity for C15 assurance."""

from __future__ import annotations

import base64
import csv
import io
import json
import tarfile
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import NotRequired, TypedDict, cast

_SCHEMA_VERSION = "1.1.0"
_NORMALIZATION = "sorted-members+safe-utf8-text-lf+content-sha256+record-semantics-v3"
_ALLOWED_TEXT_CONTROLS = frozenset("\t\n\r")
_MEMBER_TYPES = frozenset(
    {
        "file",
        "directory",
        "symlink",
        "hardlink",
        "character-device",
        "block-device",
        "fifo",
    }
)


class ArtifactMismatch(ValueError):
    """Raised when normalized artifacts do not have identical content."""


class ArchiveEntry(TypedDict):
    """Normalized content identity for one archive member."""

    path: str
    member_type: str
    sha256: str
    size: int
    link_target: NotRequired[str]
    device_major: NotRequired[int]
    device_minor: NotRequired[int]


def _safe_path(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path.as_posix()


def _normalized_content(content: bytes) -> bytes:
    """Normalize line endings only when bytes are unambiguously safe UTF-8 text."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    if any(
        (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
        and character not in _ALLOWED_TEXT_CONTROLS
        for character in text
    ):
        return content
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _zip_entries(path: Path) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.is_dir():
                entries.append((member.filename, archive.read(member)))
    return entries


def _metadata_entry(
    path: str,
    member_type: str,
    *,
    link_target: str | None = None,
    device_major: int | None = None,
    device_minor: int | None = None,
) -> ArchiveEntry:
    metadata: dict[str, object] = {"member_type": member_type}
    entry: ArchiveEntry = {
        "path": path,
        "member_type": member_type,
        "sha256": "",
        "size": 0,
    }
    if link_target is not None:
        metadata["link_target"] = link_target
        entry["link_target"] = link_target
    if device_major is not None and device_minor is not None:
        metadata["device_major"] = device_major
        metadata["device_minor"] = device_minor
        entry["device_major"] = device_major
        entry["device_minor"] = device_minor
    entry["sha256"] = sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return entry


def _tar_entries(path: Path) -> tuple[list[tuple[str, bytes]], list[ArchiveEntry]]:
    regular: list[tuple[str, bytes]] = []
    metadata: list[ArchiveEntry] = []
    seen: set[str] = set()
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            name = _safe_path(member.name)
            if name in seen:
                raise ValueError(f"duplicate archive path: {name}")
            seen.add(name)
            if member.isdir():
                metadata.append(_metadata_entry(name, "directory"))
                continue
            if member.issym() or member.islnk():
                target = _safe_path(member.linkname)
                member_type = "symlink" if member.issym() else "hardlink"
                metadata.append(_metadata_entry(name, member_type, link_target=target))
                continue
            if member.ischr() or member.isblk():
                member_type = "character-device" if member.ischr() else "block-device"
                metadata.append(
                    _metadata_entry(
                        name,
                        member_type,
                        device_major=member.devmajor,
                        device_minor=member.devminor,
                    )
                )
                continue
            if member.isfifo():
                metadata.append(_metadata_entry(name, "fifo"))
                continue
            if not member.isfile():
                raise ValueError(f"unsupported tar member type: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member cannot be read: {member.name}")
            regular.append((name, extracted.read()))
    return regular, metadata


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
        "member_type": "file",
        "sha256": sha256(content).hexdigest(),
        "size": len(content),
    }


def _raw_archive(path: Path) -> tuple[dict[str, bytes], list[ArchiveEntry]]:
    if zipfile.is_zipfile(path):
        raw_entries = _zip_entries(path)
        metadata: list[ArchiveEntry] = []
    elif tarfile.is_tarfile(path):
        raw_entries, metadata = _tar_entries(path)
    else:
        raise ValueError(f"unsupported archive format: {path.name}")
    raw: dict[str, bytes] = {}
    for raw_name, raw_content in raw_entries:
        name = _safe_path(raw_name)
        if name in raw:
            raise ValueError(f"duplicate archive path: {name}")
        raw[name] = raw_content
    return raw, metadata


def _entries(path: Path) -> list[ArchiveEntry]:
    raw, metadata = _raw_archive(path)
    record_names = [name for name in raw if name.endswith(".dist-info/RECORD")]
    if path.suffix == ".whl" and len(record_names) != 1:
        raise ValueError("wheel must contain exactly one .dist-info/RECORD")
    if len(record_names) > 1:
        raise ValueError("archive contains multiple .dist-info/RECORD files")

    entries = metadata
    for name, raw_content in raw.items():
        if name in record_names:
            continue
        content = _normalized_content(raw_content)
        entries.append(
            {
                "path": name,
                "member_type": "file",
                "sha256": sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    entries.sort(key=lambda item: item["path"])
    if record_names:
        entries.append(_wheel_record_entry(raw, record_names[0], entries))
    if not raw:
        raise ValueError("archive contains no regular files")
    return sorted(entries, key=lambda item: item["path"])


def _artifact_kind(path: Path) -> str:
    if path.suffix == ".whl":
        return "wheel"
    if path.name.endswith(".tar.gz"):
        return "sdist"
    if zipfile.is_zipfile(path):
        return "zip"
    return "tar"


def _canonical_entries(entries: list[ArchiveEntry]) -> bytes:
    return json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()


def normalized_archive_report(path: Path, *, runner: str) -> dict[str, object]:
    """Return metadata-independent content identity for a zip, wheel, or tar archive."""
    if not runner.strip():
        raise ValueError("runner identity must not be empty")
    entries = _entries(path)
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_name": path.name,
        "artifact_kind": _artifact_kind(path),
        "runner": runner,
        "normalization": _NORMALIZATION,
        "normalized_sha256": sha256(_canonical_entries(entries)).hexdigest(),
        "entries": entries,
    }


def _lower_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_identity_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return value if _safe_path(value) == value else None
    except ValueError:
        return None


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _entry_expected_keys(
    label: str, entry: dict[str, object], member_type: str
) -> set[str]:
    expected = {"path", "member_type", "sha256", "size"}
    if member_type in {"symlink", "hardlink"}:
        if _safe_identity_path(entry.get("link_target")) is None:
            raise ArtifactMismatch(f"{label} archive link target is invalid")
        expected.add("link_target")
    if member_type in {"character-device", "block-device"}:
        if not all(
            _nonnegative_integer(entry.get(key))
            for key in ("device_major", "device_minor")
        ):
            raise ArtifactMismatch(f"{label} archive device is invalid")
        expected.update({"device_major", "device_minor"})
    return expected


def _validated_entry(label: str, raw: object) -> ArchiveEntry:
    if not isinstance(raw, dict):
        raise ArtifactMismatch(f"{label} archive entry is invalid")
    entry = cast("dict[str, object]", raw)
    path = _safe_identity_path(entry.get("path"))
    member_type = entry.get("member_type")
    if (
        path is None
        or not isinstance(member_type, str)
        or member_type not in _MEMBER_TYPES
        or not _lower_hex(entry.get("sha256"))
        or not _nonnegative_integer(entry.get("size"))
    ):
        raise ArtifactMismatch(f"{label} archive entry is invalid")
    if set(entry) != _entry_expected_keys(label, entry, member_type):
        raise ArtifactMismatch(f"{label} archive entry schema is invalid")
    if member_type != "file":
        expected = _metadata_entry(
            path,
            member_type,
            link_target=cast("str | None", entry.get("link_target")),
            device_major=cast("int | None", entry.get("device_major")),
            device_minor=cast("int | None", entry.get("device_minor")),
        )
        if entry != expected:
            raise ArtifactMismatch(f"{label} archive metadata digest is invalid")
    return cast("ArchiveEntry", entry)


def _validated_entries(label: str, value: object) -> list[ArchiveEntry]:
    if not isinstance(value, list) or not value:
        raise ArtifactMismatch(f"{label} archive inventory is invalid")
    entries: list[ArchiveEntry] = []
    paths: list[str] = []
    for raw in value:
        entry = _validated_entry(label, raw)
        paths.append(entry["path"])
        entries.append(entry)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ArtifactMismatch(f"{label} archive paths must be unique and sorted")
    return entries


def _validate_report(label: str, report: dict[str, object]) -> list[ArchiveEntry]:
    required_keys = {
        "schema_version",
        "artifact_name",
        "artifact_kind",
        "runner",
        "normalization",
        "normalized_sha256",
        "entries",
    }
    if set(report) != required_keys:
        raise ArtifactMismatch(f"{label} digest report schema is invalid")
    if report.get("schema_version") != _SCHEMA_VERSION:
        raise ArtifactMismatch(f"{label} digest report schema is unsupported")
    if report.get("normalization") != _NORMALIZATION:
        raise ArtifactMismatch(f"{label} normalization policy is unsupported")
    name = report.get("artifact_name")
    if (
        not isinstance(name, str)
        or not name
        or PurePosixPath(name.replace("\\", "/")).name != name
    ):
        raise ArtifactMismatch(f"{label} artifact name is invalid")
    kind = report.get("artifact_kind")
    if not isinstance(kind, str) or kind not in {"wheel", "sdist", "zip", "tar"}:
        raise ArtifactMismatch(f"{label} artifact kind is invalid")
    kind_matches_name = (
        (kind == "wheel" and name.endswith(".whl"))
        or (kind == "sdist" and name.endswith(".tar.gz"))
        or (kind == "zip" and name.endswith(".zip"))
        or (kind == "tar" and name.endswith((".tar", ".tar.bz2", ".tar.xz")))
    )
    if not kind_matches_name:
        raise ArtifactMismatch(f"{label} artifact kind does not match its name")
    runner = report.get("runner")
    if not isinstance(runner, str) or not runner.strip():
        raise ArtifactMismatch(f"{label} runner identity is invalid")
    entries = _validated_entries(label, report.get("entries"))
    digest = report.get("normalized_sha256")
    if (
        not _lower_hex(digest)
        or digest != sha256(_canonical_entries(entries)).hexdigest()
    ):
        raise ArtifactMismatch(f"{label} normalized digest is invalid")
    return entries


def compare_digest_reports(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    """Fail closed unless two independently generated reports have equal identity."""
    left_entries = _validate_report("left", left)
    right_entries = _validate_report("right", right)
    if left.get("runner") == right.get("runner"):
        raise ArtifactMismatch("independent runner identities must differ")
    if left.get("artifact_kind") != right.get("artifact_kind"):
        raise ArtifactMismatch("artifact kinds differ")
    if left.get("artifact_name") != right.get("artifact_name"):
        raise ArtifactMismatch("artifact names differ")
    if left_entries != right_entries:
        raise ArtifactMismatch("normalized archive inventories differ")
    return {
        "schema_version": _SCHEMA_VERSION,
        "matched": True,
        "artifact_kind": left["artifact_kind"],
        "artifact_name": left["artifact_name"],
        "normalized_sha256": left["normalized_sha256"],
        "runners": [left.get("runner"), right.get("runner")],
    }


__all__ = [
    "ArtifactMismatch",
    "compare_digest_reports",
    "normalized_archive_report",
]
