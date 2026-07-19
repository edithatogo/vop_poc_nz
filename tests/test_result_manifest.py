from __future__ import annotations

import json
from pathlib import Path

from vop_poc_nz.result_manifest import build_result_manifest, sha256_file


def test_sha256_file_and_manifest(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "figure.png"
    input_file.write_text("a,b\n1,2\n", encoding="utf-8")
    output_file.write_bytes(b"fake png")

    hashed = sha256_file(input_file)
    assert hashed.sha256
    assert hashed.bytes == input_file.stat().st_size

    manifest = build_result_manifest(
        run_id="housing-smoke",
        script="scripts/make_figure.py",
        inputs=[input_file],
        outputs=[output_file],
        case_id="housing_insulation_nz",
        random_seed=123,
        parameters={"threshold": 50_000},
    )
    payload = manifest.as_dict()
    assert payload["case_id"] == "housing_insulation_nz"
    assert payload["outputs"][0]["sha256"]
    manifest_path = manifest.write_json(tmp_path / "manifest.json")
    assert json.loads(manifest_path.read_text())["run_id"] == "housing-smoke"
