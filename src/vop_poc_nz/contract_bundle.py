"""Deterministic producer and verifier for the shared VOP-VOIAGE contract bundle."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from vop_poc_nz.perspective_io import attach_contract_metadata, schema_fingerprint
from vop_poc_nz.pipeline.typed import TYPED_PIPELINE_ARROW_SCHEMA

BUNDLE_ID = "vop-voiage-contracts"
BUNDLE_VERSION = "1.0.0"
CONTRACT_VERSION = "1.0.0"
METHOD_CONTRACT_VERSION = "1.1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FIXTURE_RECORDS: tuple[dict[str, object], ...] = (
    {
        "run_id": "bundle-fixture-v1",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "software_version": "1.0.0",
        "random_seed": 20260720,
        "spec_fingerprint": "0" * 64,
        "contract_version": CONTRACT_VERSION,
        "intervention": "usual-care",
        "perspective": "health_system",
        "productivity_cost_method": "human_capital",
        "incremental_cost": 0.0,
        "incremental_qalys": 0.0,
        "incremental_nmb": 0.0,
        "icer_status": "undefined",
        "icer_value": None,
        "is_cost_effective": True,
        "wtp_threshold": 50_000.0,
        "cost_unit": "NZD",
        "cost_currency_code": "NZD",
        "cost_currency_year": 2026,
        "health_outcome_unit": "QALY",
    },
    {
        "run_id": "bundle-fixture-v1",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "software_version": "1.0.0",
        "random_seed": 20260720,
        "spec_fingerprint": "0" * 64,
        "contract_version": CONTRACT_VERSION,
        "intervention": "new-treatment",
        "perspective": "societal",
        "productivity_cost_method": "human_capital",
        "incremental_cost": 1250.5,
        "incremental_qalys": 0.125,
        "incremental_nmb": 4999.5,
        "icer_status": "finite",
        "icer_value": 10_004.0,
        "is_cost_effective": True,
        "wtp_threshold": 50_000.0,
        "cost_unit": "NZD",
        "cost_currency_code": "NZD",
        "cost_currency_year": 2026,
        "health_outcome_unit": "QALY",
    },
)

_UNITS: dict[str, object] = {
    "incremental_cost": {"symbol_field": "cost_unit", "dimension": "currency"},
    "incremental_nmb": {"symbol_field": "cost_unit", "dimension": "currency"},
    "wtp_threshold": {"symbol_field": "cost_unit", "dimension": "currency_per_health"},
    "incremental_qalys": {"symbol_field": "health_outcome_unit", "dimension": "health"},
    "icer_value": {"symbol_field": "cost_unit", "dimension": "currency_per_health"},
}


class IncompatibleContractChange(ValueError):
    """Raised when schema evolution is not permitted by the shared policy."""


def _semantic_version(value: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value) is None
    ):
        raise IncompatibleContractChange("schema version is invalid")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def _validate_fingerprint_transition(
    previous: object, current: object, *, changed: bool
) -> None:
    if (
        not isinstance(previous, str)
        or not isinstance(current, str)
        or _SHA256_RE.fullmatch(previous) is None
        or _SHA256_RE.fullmatch(current) is None
        or (changed == (previous == current))
    ):
        raise IncompatibleContractChange("schema fingerprint transition is invalid")


def _added_field_names(additions: Sequence[object]) -> list[str]:
    names: list[str] = []
    for addition in additions:
        if not isinstance(addition, dict):
            raise IncompatibleContractChange("new fields must be nullable")
        name = addition.get("name")
        if addition.get("nullable") is not True or not isinstance(name, str):
            raise IncompatibleContractChange("new fields must be nullable")
        names.append(name)
    return names


def canonical_json_bytes(value: object) -> bytes:
    """Encode canonical UTF-8 JSON used by all bundle digests."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def arrow_identity_document(
    schema: pa.Schema = TYPED_PIPELINE_ARROW_SCHEMA,
) -> dict[str, object]:
    """Describe the exact cross-language Arrow identity and semantic units."""
    return {
        "schema_id": "typed_pipeline_records",
        "schema_version": CONTRACT_VERSION,
        "schema_fingerprint": schema_fingerprint(schema),
        "fields": [
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "nullable": field.nullable,
                "unit": _UNITS.get(field.name),
            }
            for field in schema
        ],
        "required_metadata": [
            "vop_voiage.contract_version",
            "vop_voiage.interchange",
            "vop_voiage.method_contract_version",
            "vop_voiage.producer",
            "vop_voiage.provenance_json",
            "vop_voiage.provenance_sha256",
            "vop_voiage.schema_fingerprint",
            "vop_voiage.schema_id",
            "vop_voiage.schema_version",
        ],
    }


def migration_policy_document() -> dict[str, object]:
    """Return the fail-closed v1 schema migration policy."""
    return {
        "schema_version": "1.0.0",
        "bundle_id": BUNDLE_ID,
        "current_bundle_version": BUNDLE_VERSION,
        "integrity": "sha256-manifest; unsigned until approved release publication",
        "compatible_changes": ["append_nullable_field"],
        "incompatible_changes": [
            "change_dtype",
            "change_nullability",
            "change_provenance_requirement",
            "change_schema_id",
            "change_semantic_unit",
            "insert_or_reorder_field",
            "remove_field",
            "unknown_change",
        ],
        "consumer_rule": "reject every change not explicitly compatible",
    }


def assess_arrow_evolution(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> dict[str, object]:
    """Assess an identity transition, rejecting all undeclared incompatibility."""
    if previous.get("schema_id") != current.get("schema_id"):
        raise IncompatibleContractChange("schema identity changed")
    previous_version = _semantic_version(previous.get("schema_version"))
    current_version = _semantic_version(current.get("schema_version"))
    if current_version[0] != previous_version[0] or current_version < previous_version:
        raise IncompatibleContractChange("schema version is incompatible")
    previous_fields = previous.get("fields")
    current_fields = current.get("fields")
    if not isinstance(previous_fields, list) or not isinstance(current_fields, list):
        raise IncompatibleContractChange("schema fields must be ordered arrays")
    if len(current_fields) < len(previous_fields):
        raise IncompatibleContractChange("schema fields were removed")
    if current_fields[: len(previous_fields)] != previous_fields:
        raise IncompatibleContractChange(
            "existing dtype, unit, nullability, or order changed"
        )
    additions = current_fields[len(previous_fields) :]
    added_fields = _added_field_names(additions)
    _validate_fingerprint_transition(
        previous.get("schema_fingerprint"),
        current.get("schema_fingerprint"),
        changed=bool(additions),
    )
    if additions and current_version == previous_version:
        raise IncompatibleContractChange("additive schemas require a version increment")
    previous_metadata = previous.get("required_metadata")
    current_metadata = current.get("required_metadata")
    if previous_metadata != current_metadata:
        raise IncompatibleContractChange("provenance or identity metadata changed")
    return {
        "backward_compatible": True,
        "forward_compatible": not additions,
        "added_fields": added_fields,
    }


def _write_fixture(output: Path) -> None:
    provenance = (
        canonical_json_bytes(
            [
                {
                    "source_id": "fixture:typed-pipeline-records-v1",
                    "metadata_status": "known",
                }
            ]
        )
        .decode("utf-8")
        .rstrip("\n")
    )
    table = pa.Table.from_pylist(
        list(_FIXTURE_RECORDS), schema=TYPED_PIPELINE_ARROW_SCHEMA
    )
    table = attach_contract_metadata(
        table,
        schema_id="typed_pipeline_records",
        schema_version=CONTRACT_VERSION,
        contract_version=CONTRACT_VERSION,
        method_contract_version=METHOD_CONTRACT_VERSION,
        expected_fingerprint=schema_fingerprint(TYPED_PIPELINE_ARROW_SCHEMA),
        provenance_json=provenance,
    )
    fixture = output / "fixtures"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "typed-pipeline-records.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_id": "typed_pipeline_records",
                "schema_fingerprint": schema_fingerprint(table.schema),
                "records": _FIXTURE_RECORDS,
            }
        )
    )
    with ipc.new_file(fixture / "typed-pipeline-records.arrow", table.schema) as writer:
        writer.write_table(table)
    pq.write_table(
        table,
        fixture / "typed-pipeline-records.parquet",
        compression="zstd",
        version="2.6",
        write_statistics=True,
    )


def _media_type(path: Path) -> str:
    return {
        ".arrow": "application/vnd.apache.arrow.file",
        ".json": "application/json",
        ".parquet": "application/vnd.apache.parquet",
    }[path.suffix]


def _file_entries(output: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        content = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256(content).hexdigest(),
                "size": len(content),
                "media_type": _media_type(path),
            }
        )
    return entries


def export_contract_bundle(output: Path, schema_source: Path) -> Path:
    """Generate a byte-reproducible versioned bundle and SHA-256 manifest."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    schemas = output / "schemas"
    schemas.mkdir()
    schema_paths = sorted(schema_source.glob("*.schema.json"))
    if not schema_paths:
        raise ValueError("canonical domain schemas are missing")
    for schema_path in schema_paths:
        shutil.copyfile(schema_path, schemas / schema_path.name)
    arrow = output / "arrow"
    arrow.mkdir()
    (arrow / "typed-pipeline-records.schema.json").write_bytes(
        canonical_json_bytes(arrow_identity_document())
    )
    (output / "migration-policy.json").write_bytes(
        canonical_json_bytes(migration_policy_document())
    )
    _write_fixture(output)
    entries = _file_entries(output)
    manifest = {
        "schema_version": "1.0.0",
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "method_contract_version": METHOD_CONTRACT_VERSION,
        "producer": "vop_poc_nz",
        "source_repository": "edithatogo/vop_poc_nz",
        "source_revision": f"contract-bundle/{BUNDLE_VERSION}",
        "source_path": f"contracts/vop-voiage/{BUNDLE_VERSION}",
        "integrity_algorithm": "sha256",
        "signature_policy": "unsigned_sha256_manifest",
        "schema_source": schema_source.as_posix(),
        "files": entries,
        "bundle_sha256": sha256(canonical_json_bytes(entries)).hexdigest(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path


def _verify_file_entry(output: Path, entry: object, declared: set[str]) -> str:
    if not isinstance(entry, dict):
        raise ValueError("contract bundle file entry must be an object")
    relative = entry.get("path")
    digest = entry.get("sha256")
    size = entry.get("size")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError("contract bundle file identity is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError("contract bundle path is unsafe")
    if relative in declared or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("contract bundle file identity is invalid")
    path = output.joinpath(*pure.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
        raise ValueError("contract bundle path is unsafe")
    content = path.read_bytes()
    if type(size) is not int or size != len(content):
        raise ValueError(f"contract bundle size mismatch: {relative}")
    if sha256(content).hexdigest() != digest:
        raise ValueError(f"contract bundle digest mismatch: {relative}")
    return relative


def verify_contract_bundle(output: Path) -> dict[str, Any]:
    """Verify exact file inventory, safe paths, byte sizes, and all digests."""
    raw = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    required_identity = {
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "method_contract_version": METHOD_CONTRACT_VERSION,
        "producer": "vop_poc_nz",
        "source_repository": "edithatogo/vop_poc_nz",
        "source_revision": f"contract-bundle/{BUNDLE_VERSION}",
        "source_path": f"contracts/vop-voiage/{BUNDLE_VERSION}",
        "integrity_algorithm": "sha256",
        "signature_policy": "unsigned_sha256_manifest",
    }
    if not isinstance(raw, dict) or any(
        raw.get(field) != value for field, value in required_identity.items()
    ):
        raise ValueError("unsupported contract bundle manifest")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("contract bundle manifest requires files")
    declared: set[str] = set()
    for entry in files:
        declared.add(_verify_file_entry(output, entry, declared))
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != declared:
        raise ValueError("contract bundle file inventory mismatch")
    expected_bundle = sha256(canonical_json_bytes(files)).hexdigest()
    if raw.get("bundle_sha256") != expected_bundle:
        raise ValueError("contract bundle aggregate digest mismatch")
    return raw


__all__ = [
    "BUNDLE_ID",
    "BUNDLE_VERSION",
    "IncompatibleContractChange",
    "arrow_identity_document",
    "assess_arrow_evolution",
    "canonical_json_bytes",
    "export_contract_bundle",
    "migration_policy_document",
    "verify_contract_bundle",
]
