"""Tests for deterministic build-evidence comparison."""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.reproducible_build import artifact_evidence, compare_build_directories


def _write_wheel(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("package/module.py", payload)


def test_identical_build_artifacts_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    artifact = first / "package-1.0-py3-none-any.whl"
    _write_wheel(artifact, b"value = 1\n")
    shutil.copyfile(artifact, second / artifact.name)
    source = first / "package-1.0.tar.gz"
    with tarfile.open(source, "w:gz"):
        pass
    shutil.copyfile(source, second / source.name)

    report = compare_build_directories(first, second)

    assert report["reproducible"] is True
    assert report["artifact_set_complete"] is True
    assert report["artifacts"][0]["byte_identical"] is True


def test_inventory_identity_does_not_hide_byte_drift(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = "package-1.0-py3-none-any.whl"
    _write_wheel(first / name, b"value = 1\n")
    _write_wheel(second / name, b"value = 2\n")
    for root in (first, second):
        with tarfile.open(root / "package-1.0.tar.gz", "w:gz"):
            pass

    report = compare_build_directories(first, second)

    assert report["reproducible"] is False
    assert report["artifacts"][0]["inventory_identical"] is False


def test_missing_sdist_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        _write_wheel(root / "package-1.0-py3-none-any.whl", b"value = 1\n")

    report = compare_build_directories(first, second)

    assert report["artifact_set_complete"] is False
    assert report["reproducible"] is False


def test_non_archive_artifact_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "package.bin"
    artifact.write_bytes(b"opaque")

    with pytest.raises(ValueError, match="unsupported build artifact"):
        artifact_evidence(artifact)
