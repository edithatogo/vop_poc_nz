from __future__ import annotations

import json
from pathlib import Path

from vop_poc_nz.evidence_manifest import (
    build_evidence_manifest,
    verify_evidence_manifest,
    write_evidence_manifest,
)


def test_manifest_normalises_windows_and_unix_line_endings(tmp_path: Path) -> None:
    unix = tmp_path / "unix.md"
    windows = tmp_path / "windows.md"
    unix.write_bytes(b"one\ntwo\n")
    windows.write_bytes(b"one\r\ntwo\r\n")

    entries = build_evidence_manifest([windows, unix], root=tmp_path)["files"]

    assert entries[0]["sha256"] == entries[1]["sha256"]
    assert entries[0]["normalization"] == "utf8-lf"


def test_manifest_is_stable_and_verifiable(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"value": 1}\n', encoding="utf-8")
    first = build_evidence_manifest([evidence], root=tmp_path)
    second = build_evidence_manifest([evidence], root=tmp_path)
    assert first == second

    path = write_evidence_manifest(first, tmp_path / "manifest.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert verify_evidence_manifest(loaded, root=tmp_path) == []

    evidence.write_text('{"value": 2}\n', encoding="utf-8")
    assert verify_evidence_manifest(loaded, root=tmp_path) == [
        "evidence.json: sha256 mismatch"
    ]
