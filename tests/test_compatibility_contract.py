from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_compatibility_contract import CONTRACT, validate


def test_shared_compatibility_contract_matches_vop_runtime() -> None:
    contract = validate()
    assert contract["profiles"]["net_benefit_records"]["producer"] == "vop_poc_nz"
    assert contract["profiles"]["directional_regret"]["producer"] == "voiage"


def test_contract_schema_is_strict_and_versioned() -> None:
    schema_path = CONTRACT.parent.parent / "compatibility-contract.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["additionalProperties"] is False
    assert Path(CONTRACT).is_file()
