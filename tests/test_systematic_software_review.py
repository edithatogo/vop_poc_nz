from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path("scripts/systematic_software_review.py")
    spec = importlib.util.spec_from_file_location("systematic_software_review", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_flow_and_records_are_consistent() -> None:
    module = _module()
    payload = json.loads(
        Path("manuscript/software_review/systematic-software-review.json").read_text(
            encoding="utf-8"
        )
    )
    assert module.validate(payload) == []
    assert payload["flow"]["included"] == 10
    assert all(item["named_directional_vop"] == "No" for item in payload["included"])


def test_generated_review_is_manifest_backed() -> None:
    module = _module()
    module.generate(Path.cwd())
    assert module.verify(Path.cwd()) == []
    tex = Path("manuscript/software_review/software-comparison.tex").read_text(
        encoding="utf-8"
    )
    assert "dceasimR" in tex
    assert "heormodel" in tex
