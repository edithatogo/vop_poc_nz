"""Validate the shared VOP-VOIAGE runtime and interchange contract."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/vop-voiage/compatibility/v1/contract.json"


def validate() -> dict[str, object]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert contract["contract_version"].count(".") == 2
    assert contract["consumers"] == ["vop_poc_nz", "voiage"]
    assert project["project"]["requires-python"] == contract["runtime"]["python"]
    declared = project["project"]["dependencies"]
    for package, required in contract["shared_dependencies"].items():
        assert any(
            item.lower().startswith(package) and required in item for item in declared
        ), package
    fingerprint = contract["interchange"]["fingerprint"]
    assert fingerprint["algorithm"] == "sha256"
    assert fingerprint["field_order"] == "preserved"
    return contract


if __name__ == "__main__":
    validated = validate()
    print(f"validated compatibility contract {validated['contract_version']}")
