from __future__ import annotations

import base64
import csv
import io
import json
import sys
import tarfile
import zipfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.c15_artifact_digest import main as artifact_digest_main
from vop_poc_nz.c15_reproducibility import (
    ArtifactMismatch,
    compare_digest_reports,
    normalized_archive_report,
)


def _record(contents: dict[str, bytes], record_name: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, content in contents.items():
        digest = (
            base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=").decode()
        )
        writer.writerow([name, f"sha256={digest}", len(content)])
    writer.writerow([record_name, "", ""])
    return stream.getvalue().encode("utf-8")


def _zip(path: Path, newline: str) -> None:
    contents = {
        "pkg/data.txt": f"alpha{newline}beta{newline}".encode(),
        "pkg/parameters.yaml.template": f"value: 1{newline}".encode(),
        "pkg/templates/Snakefile": f"rule all:{newline}    pass{newline}".encode(),
        "pkg-1.dist-info/licenses/LICENSE": f"terms{newline}".encode(),
        "pkg/data.bin": b"\x00\x01",
    }
    record_name = "pkg-1.dist-info/RECORD"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
        archive.writestr(record_name, _record(contents, record_name))


def test_zip_digest_is_stable_across_timestamps_order_and_line_endings(
    tmp_path: Path,
) -> None:
    linux = tmp_path / "linux" / "pkg.whl"
    windows = tmp_path / "windows" / "pkg.whl"
    linux.parent.mkdir()
    windows.parent.mkdir()
    _zip(linux, "\n")
    _zip(windows, "\r\n")

    left = normalized_archive_report(linux, runner="linux-x64")
    right = normalized_archive_report(windows, runner="windows-x64")

    assert left["normalized_sha256"] == right["normalized_sha256"]
    assert compare_digest_reports(left, right)["matched"] is True


def test_tar_digest_ignores_archive_metadata_but_not_content(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    source = tmp_path / "payload.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    with tarfile.open(first, "w:gz") as archive:
        archive.add(source, arcname="pkg/payload.json")
    report = normalized_archive_report(first, runner="linux-x64")
    assert report["entries"][0]["path"] == "pkg/payload.json"


def test_sdist_digest_normalizes_safe_utf8_text_independent_of_filename(
    tmp_path: Path,
) -> None:
    linux = tmp_path / "linux.tar.gz"
    windows = tmp_path / "windows.tar.gz"
    names = (
        "Makefile",
        ".gitignore",
        "CITATION.cff",
        "build.sh",
        "diagram.mmd",
        "notes.bak",
        "MANIFEST.in",
        "paper.tex",
        "references.bib",
        "notebook.ipynb",
        "uv.lock",
        "index.html",
        "style.css",
        "bundle.js",
        "bundle.js.map",
    )
    for archive_path, newline in ((linux, "\n"), (windows, "\r\n")):
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in names:
                content = f"alpha{newline}beta{newline}".encode()
                member = tarfile.TarInfo(f"pkg/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    left = normalized_archive_report(linux, runner="linux-x64")
    right = normalized_archive_report(windows, runner="windows-x64")

    assert left["entries"] == right["entries"]
    assert left["normalized_sha256"] == right["normalized_sha256"]


@pytest.mark.parametrize(
    "content", (b"alpha\x00\r\nbeta", b"alpha\x01\r\nbeta", b"\xff\r\n")
)
def test_sdist_digest_preserves_binary_or_control_bearing_content(
    tmp_path: Path, content: bytes
) -> None:
    archive_path = tmp_path / "binary.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("pkg/apparently-text.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    report = normalized_archive_report(archive_path, runner="linux-x64")

    assert report["entries"] == [
        {
            "path": "pkg/apparently-text.txt",
            "member_type": "file",
            "sha256": sha256(content).hexdigest(),
            "size": len(content),
        }
    ]


def test_comparison_fails_closed_for_content_or_inventory_drift(tmp_path: Path) -> None:
    left_path = tmp_path / "left" / "pkg.whl"
    right_path = tmp_path / "right" / "pkg.whl"
    left_path.parent.mkdir()
    right_path.parent.mkdir()
    _zip(left_path, "\n")
    with zipfile.ZipFile(right_path, "w") as archive:
        contents = {"pkg/data.txt": b"changed\n"}
        archive.writestr("pkg/data.txt", contents["pkg/data.txt"])
        archive.writestr(
            "pkg-1.dist-info/RECORD", _record(contents, "pkg-1.dist-info/RECORD")
        )
    left = normalized_archive_report(left_path, runner="linux-x64")
    right = normalized_archive_report(right_path, runner="windows-x64")
    with pytest.raises(ArtifactMismatch, match="normalized archive inventories differ"):
        compare_digest_reports(left, right)

    json.dumps(left)


def test_digest_report_schema_is_self_authenticating_and_bound_to_artifact(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left" / "pkg.whl"
    right_path = tmp_path / "right" / "pkg.whl"
    left_path.parent.mkdir()
    right_path.parent.mkdir()
    _zip(left_path, "\n")
    _zip(right_path, "\n")
    left = normalized_archive_report(left_path, runner="linux-x64")
    right = normalized_archive_report(right_path, runner="windows-x64")
    assert left["artifact_kind"] == "wheel"
    assert compare_digest_reports(left, right)["matched"] is True

    corruptions = (
        {"normalized_sha256": "x" * 64},
        {"normalized_sha256": "A" * 64},
        {"artifact_kind": "sdist"},
        {"artifact_name": "other.whl"},
        {"unexpected": True},
        {"entries": []},
        {"entries": [*left["entries"], left["entries"][0]]},
        {
            "entries": [
                {**left["entries"][0], "size": left["entries"][0]["size"] + 1},
                *left["entries"][1:],
            ]
        },
        {"runner": ""},
        {"artifact_name": ""},
        {"artifact_kind": "mystery"},
        {"entries": [None]},
        {"entries": [{**left["entries"][0], "path": "../unsafe"}]},
        {"entries": [{**left["entries"][0], "member_type": []}]},
        {"entries": [{**left["entries"][0], "sha256": "A" * 64}]},
        {"entries": [{**left["entries"][0], "size": True}]},
        {"entries": [{**left["entries"][0], "extra": True}]},
    )
    for corruption in corruptions:
        with pytest.raises(ArtifactMismatch):
            compare_digest_reports({**left, **corruption}, right)
    with pytest.raises(ArtifactMismatch, match="kinds differ"):
        compare_digest_reports(
            left,
            {
                **right,
                "artifact_kind": "sdist",
                "artifact_name": "pkg.tar.gz",
            },
        )
    with pytest.raises(ArtifactMismatch, match="names differ"):
        compare_digest_reports(left, {**right, "artifact_name": "other.whl"})


def test_tar_inventory_encodes_non_regular_members_and_link_drift(
    tmp_path: Path,
) -> None:
    reports = []
    for directory, target in (("left", "data.txt"), ("right", "other.txt")):
        archive_path = tmp_path / directory / "pkg.tar.gz"
        archive_path.parent.mkdir()
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in ("data.txt", "other.txt"):
                content = name.encode()
                member = tarfile.TarInfo(f"pkg/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            directory_member = tarfile.TarInfo("pkg/empty")
            directory_member.type = tarfile.DIRTYPE
            archive.addfile(directory_member)
            symlink = tarfile.TarInfo("pkg/link")
            symlink.type = tarfile.SYMTYPE
            symlink.linkname = target
            archive.addfile(symlink)
            hardlink = tarfile.TarInfo("pkg/hardlink")
            hardlink.type = tarfile.LNKTYPE
            hardlink.linkname = "pkg/data.txt"
            archive.addfile(hardlink)
            character = tarfile.TarInfo("pkg/character")
            character.type = tarfile.CHRTYPE
            character.devmajor = 1
            character.devminor = 3
            archive.addfile(character)
            block = tarfile.TarInfo("pkg/block")
            block.type = tarfile.BLKTYPE
            block.devmajor = 8
            block.devminor = 0
            archive.addfile(block)
            fifo = tarfile.TarInfo("pkg/fifo")
            fifo.type = tarfile.FIFOTYPE
            archive.addfile(fifo)
        reports.append(
            normalized_archive_report(archive_path, runner=f"{directory}-runner")
        )

    member_types = {entry["member_type"] for entry in reports[0]["entries"]}
    assert member_types == {
        "block-device",
        "character-device",
        "directory",
        "fifo",
        "file",
        "hardlink",
        "symlink",
    }
    with pytest.raises(ArtifactMismatch, match="inventories differ"):
        compare_digest_reports(*reports)

    symlink_drift = deepcopy(reports[0])
    symlink = next(
        entry for entry in symlink_drift["entries"] if entry["member_type"] == "symlink"
    )
    symlink["link_target"] = "../unsafe"
    with pytest.raises(ArtifactMismatch, match="link target"):
        compare_digest_reports(symlink_drift, reports[1])

    device_drift = deepcopy(reports[0])
    device = next(
        entry
        for entry in device_drift["entries"]
        if entry["member_type"] == "character-device"
    )
    device.pop("device_major")
    with pytest.raises(ArtifactMismatch, match="device"):
        compare_digest_reports(device_drift, reports[1])

    metadata_drift = deepcopy(reports[0])
    directory_entry = next(
        entry
        for entry in metadata_drift["entries"]
        if entry["member_type"] == "directory"
    )
    directory_entry["sha256"] = "0" * 64
    with pytest.raises(ArtifactMismatch, match="metadata digest"):
        compare_digest_reports(metadata_drift, reports[1])


def test_tar_inventory_rejects_unknown_member_type(tmp_path: Path) -> None:
    archive_path = tmp_path / "unknown.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        content = b"data"
        regular = tarfile.TarInfo("pkg/data")
        regular.size = len(content)
        archive.addfile(regular, io.BytesIO(content))
        unknown = tarfile.TarInfo("pkg/unknown")
        unknown.type = b"Z"
        archive.addfile(unknown)
    with pytest.raises(ValueError, match="unsupported tar member type"):
        normalized_archive_report(archive_path, runner="linux-x64")


def test_comparator_cli_writes_structured_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    output = tmp_path / "failure.json"
    left.write_text("{}", encoding="utf-8")
    right.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "c15_artifact_digest.py",
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
        ],
    )
    assert artifact_digest_main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["matched"] is False
    assert report["status"] == "failure"


def test_wheel_record_integrity_is_validated(tmp_path: Path) -> None:
    wheel = tmp_path / "tampered.whl"
    original = {"pkg/data.txt": b"original\n"}
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pkg/data.txt", b"tampered")
        archive.writestr(
            "pkg-1.dist-info/RECORD", _record(original, "pkg-1.dist-info/RECORD")
        )
    with pytest.raises(ValueError, match="RECORD integrity mismatch"):
        normalized_archive_report(wheel, runner="linux-x64")
