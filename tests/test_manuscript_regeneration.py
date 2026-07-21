from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def regeneration_module():
    path = Path("scripts/regenerate_manuscript_results.py")
    spec = importlib.util.spec_from_file_location("manuscript_regeneration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_regeneration_is_deterministic(
    tmp_path: Path, regeneration_module
) -> None:
    root = Path.cwd()
    first = regeneration_module.generate(root, seed=20260721, draws=256)
    first_payload = first.read_bytes()
    first_results = (
        root / "manuscript/generated/publication-results.json"
    ).read_bytes()
    second = regeneration_module.generate(root, seed=20260721, draws=256)
    assert second.read_bytes() == first_payload
    assert (
        root / "manuscript/generated/publication-results.json"
    ).read_bytes() == first_results


def test_intervals_and_directional_loss_are_valid(regeneration_module) -> None:
    root = Path.cwd()
    regeneration_module.generate(root, seed=7, draws=512)
    payload = json.loads(
        (root / "manuscript/generated/publication-results.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(payload["results"]) == 5
    for case in payload["results"]:
        for key in ("hs_icer", "soc_icer", "vop_per_person", "discordance_probability"):
            summary = case[key]
            assert summary["lower_95"] <= summary["mean"] <= summary["upper_95"]
            assert summary["mcse"] >= 0
        assert case["vop_per_person"]["lower_95"] >= 0


def test_verifier_rejects_artifact_drift(tmp_path: Path, regeneration_module) -> None:
    root = Path.cwd()
    regeneration_module.generate(root, seed=11, draws=128)
    tex = root / "manuscript/generated/publication-results.tex"
    original = tex.read_bytes()
    try:
        tex.write_bytes(original + b"% drift\n")
        assert any(
            "publication-results.tex" in item
            for item in regeneration_module.verify(root)
        )
    finally:
        tex.write_bytes(original)
