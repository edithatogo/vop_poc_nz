"""Reproducibility, integrity, and evolution tests for the C14 bundle."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from vop_poc_nz.contract_bundle import (
    IncompatibleContractChange,
    arrow_identity_document,
    assess_arrow_evolution,
    export_contract_bundle,
    verify_contract_bundle,
)
from vop_poc_nz.perspective_io import schema_fingerprint

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "contracts/vop-voiage/1.0.0"
MIGRATION = ROOT / "contracts/vop-voiage/migrations/1.0.0-to-1.1.0.json"


def _identity() -> dict[str, Any]:
    return cast(dict[str, Any], arrow_identity_document())


def _descriptor_fingerprint(fields: list[dict[str, Any]]) -> str:
    arrow_types = {
        "bool": pa.bool_(),
        "double": pa.float64(),
        "int64": pa.int64(),
        "string": pa.string(),
    }
    return schema_fingerprint(
        pa.schema(
            [
                pa.field(
                    field["name"],
                    arrow_types[field["arrow_type"]],
                    nullable=field["nullable"],
                )
                for field in fields
            ]
        )
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_bundle_regeneration_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_contract_bundle(first, Path("schemas/domain"))
    export_contract_bundle(second, Path("schemas/domain"))
    assert _tree(first) == _tree(second) == _tree(BUNDLE)


def test_committed_bundle_check_mode_and_manifest_are_valid() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_contract_bundle.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = verify_contract_bundle(BUNDLE)
    assert manifest["bundle_id"] == "vop-voiage-contracts"
    assert manifest["bundle_version"] == "1.0.0"
    assert manifest["producer"] == "vop_poc_nz"
    assert manifest["source_repository"] == "edithatogo/vop_poc_nz"
    assert manifest["source_revision"] == "contract-bundle/1.0.0"
    assert manifest["signature_policy"] == "unsigned_sha256_manifest"
    assert manifest["analytical_reference"]["path"] == (
        "references/analytical-reference-manifest.json"
    )
    assert len(manifest["files"]) == 16


def test_bundle_verification_fails_closed_on_tampering(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    export_contract_bundle(output, Path("schemas/domain"))
    target = output / "migration-policy.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_contract_bundle(output)


def test_arrow_and_parquet_fixtures_share_exact_identity_and_values() -> None:
    with ipc.open_file(BUNDLE / "fixtures/typed-pipeline-records.arrow") as reader:
        arrow = reader.read_all()
    parquet = pq.read_table(BUNDLE / "fixtures/typed-pipeline-records.parquet")
    expected = json.loads(
        (BUNDLE / "fixtures/typed-pipeline-records.json").read_text(encoding="utf-8")
    )
    assert arrow.equals(parquet, check_metadata=True)
    assert arrow.to_pylist() == expected["records"]
    assert schema_fingerprint(arrow.schema) == expected["schema_fingerprint"]
    metadata = arrow.schema.metadata or {}
    assert metadata[b"vop_voiage.provenance_json"]
    assert metadata[b"vop_voiage.provenance_sha256"]
    fixture_metadata = json.loads(
        (BUNDLE / "fixtures/typed-pipeline-records.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected["fixture_binding"] == "language_runtime_neutral"
    assert fixture_metadata["binding"] == "language_runtime_neutral"
    assert metadata[b"vop_voiage.fixture_id"] == b"typed-pipeline-records-v1"
    assert (
        metadata[b"vop_voiage.fixture_metadata_sha256"].decode()
        == expected["fixture_metadata_sha256"]
    )


def test_identical_and_nullable_additive_schema_evolution_is_declared() -> None:
    previous = _identity()
    assert assess_arrow_evolution(previous, deepcopy(previous)) == {
        "backward_compatible": True,
        "forward_compatible": True,
        "added_fields": [],
    }
    current = deepcopy(previous)
    current["fields"].append(
        {
            "name": "optional_note",
            "arrow_type": "string",
            "nullable": True,
            "unit": None,
        }
    )
    current["schema_version"] = "1.1.0"
    current["schema_fingerprint"] = _descriptor_fingerprint(current["fields"])
    report = assess_arrow_evolution(previous, current)
    assert report["backward_compatible"] is True
    assert report["forward_compatible"] is False


def test_committed_previous_current_migration_matches_producer_identity() -> None:
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    assert migration["previous"] == _identity()

    report = assess_arrow_evolution(migration["previous"], migration["current"])

    assert report == {
        "backward_compatible": True,
        "forward_compatible": False,
        "added_fields": ["decision_context"],
    }


@pytest.mark.parametrize(
    "change", ["dtype", "unit", "identity", "provenance", "version"]
)
def test_incompatible_schema_evolution_fails_closed(change: str) -> None:
    previous = _identity()
    current = deepcopy(previous)
    if change == "dtype":
        current["fields"][9]["arrow_type"] = "float32"
    elif change == "unit":
        current["fields"][9]["unit"] = {"dimension": "count"}
    elif change == "identity":
        current["schema_id"] = "different"
    elif change == "version":
        current["schema_version"] = "2.0.0"
    else:
        current["required_metadata"] = []
    with pytest.raises(IncompatibleContractChange):
        assess_arrow_evolution(previous, current)


@pytest.mark.parametrize("document", ["previous", "current"])
def test_unknown_top_level_schema_semantics_fail_closed(document: str) -> None:
    previous = _identity()
    current = deepcopy(previous)
    target = previous if document == "previous" else current
    target["consumer_hint"] = "silently-ignore-me"
    with pytest.raises(IncompatibleContractChange, match="unknown top-level"):
        assess_arrow_evolution(previous, current)


@pytest.mark.parametrize("collision", ["existing", "duplicate_append"])
def test_appended_field_name_collisions_fail_closed(collision: str) -> None:
    previous = _identity()
    current = deepcopy(previous)
    appended_name = previous["fields"][0]["name"] if collision == "existing" else "note"
    current["fields"].append(
        {"name": appended_name, "arrow_type": "string", "nullable": True, "unit": None}
    )
    if collision == "duplicate_append":
        current["fields"].append(
            {"name": "note", "arrow_type": "string", "nullable": True, "unit": None}
        )
    current["schema_version"] = "1.1.0"
    current["schema_fingerprint"] = "1" * 64
    with pytest.raises(IncompatibleContractChange, match="field names collide"):
        assess_arrow_evolution(previous, current)
