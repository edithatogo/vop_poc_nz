from __future__ import annotations

import base64
import csv
import io
import json
import tarfile
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

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
    linux = tmp_path / "linux.whl"
    windows = tmp_path / "windows.whl"
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
            "sha256": sha256(content).hexdigest(),
            "size": len(content),
        }
    ]


def test_comparison_fails_closed_for_content_or_inventory_drift(tmp_path: Path) -> None:
    left_path = tmp_path / "left.whl"
    right_path = tmp_path / "right.whl"
    _zip(left_path, "\n")
    with zipfile.ZipFile(right_path, "w") as archive:
        contents = {"pkg/data.txt": b"changed\n"}
        archive.writestr("pkg/data.txt", contents["pkg/data.txt"])
        archive.writestr(
            "pkg-1.dist-info/RECORD", _record(contents, "pkg-1.dist-info/RECORD")
        )
    left = normalized_archive_report(left_path, runner="linux-x64")
    right = normalized_archive_report(right_path, runner="windows-x64")
    with pytest.raises(ArtifactMismatch, match="normalized artifact digests differ"):
        compare_digest_reports(left, right)

    json.dumps(left)


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
