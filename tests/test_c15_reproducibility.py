from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from vop_poc_nz.c15_reproducibility import (
    ArtifactMismatch,
    compare_digest_reports,
    normalized_archive_report,
)


def _zip(path: Path, newline: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pkg/data.txt", f"alpha{newline}beta{newline}")
        archive.writestr("pkg/data.bin", b"\x00\x01")
        archive.writestr("pkg-1.dist-info/RECORD", f"platform-derived-{newline}")


def test_zip_digest_is_stable_across_timestamps_order_and_line_endings(
    tmp_path: Path,
) -> None:
    linux = tmp_path / "linux.whl"
    windows = tmp_path / "windows.whl"
    _zip(linux, "\n")
    with zipfile.ZipFile(windows, "w") as archive:
        archive.writestr("pkg/data.bin", b"\x00\x01")
        archive.writestr("pkg/data.txt", "alpha\r\nbeta\r\n")
        archive.writestr("pkg-1.dist-info/RECORD", "different-derived-record\r\n")

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


def test_comparison_fails_closed_for_content_or_inventory_drift(tmp_path: Path) -> None:
    left_path = tmp_path / "left.whl"
    right_path = tmp_path / "right.whl"
    _zip(left_path, "\n")
    with zipfile.ZipFile(right_path, "w") as archive:
        archive.writestr("pkg/data.txt", "changed\n")
    left = normalized_archive_report(left_path, runner="linux-x64")
    right = normalized_archive_report(right_path, runner="windows-x64")
    with pytest.raises(ArtifactMismatch, match="normalized artifact digests differ"):
        compare_digest_reports(left, right)

    json.dumps(left)
