"""Deterministic producer and verifier for the shared VOP-VOIAGE contract bundle."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from vop_poc_nz.assurance_policy import (
    has_exact_keys,
    has_name_collision,
    matches_computed_identity,
)
from vop_poc_nz.perspective_io import attach_contract_metadata, schema_fingerprint
from vop_poc_nz.pipeline.typed import TYPED_PIPELINE_ARROW_SCHEMA

BUNDLE_ID = "vop-voiage-contracts"
BUNDLE_VERSION = "1.0.0"
CONTRACT_VERSION = "1.0.0"
METHOD_CONTRACT_VERSION = "1.1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARROW_IDENTITY_KEYS = frozenset(
    {
        "fields",
        "required_metadata",
        "schema_fingerprint",
        "schema_id",
        "schema_version",
    }
)

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


def analytical_reference_document() -> dict[str, object]:
    """Return hand-derived, implementation-independent policy reference cases."""
    return {
        "schema_version": "1.0.0",
        "reference_id": "vop-voiage-analytical-reference",
        "reference_version": "1.0.0",
        "method_contract_version": METHOD_CONTRACT_VERSION,
        "binding": "language_runtime_neutral",
        "provenance": {
            "source_id": "manual-algebra:vop-voiage-reference-v1",
            "derivation_method": "hand_calculated_closed_form",
            "implementation_dependency": "none",
            "review_status": "producer_authored_pending_external_replication",
            "derivation_date": "2026-07-20",
        },
        "comparison": {
            "rule": "abs(actual-expected) <= atol + rtol*abs(expected)",
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "boolean_comparison": "exact",
            "string_comparison": "exact_utf8",
        },
        "assumptions": [
            "Costs and health outcomes accrue at the start of each cycle.",
            "Expected values use an equally weighted arithmetic mean over draws.",
            "A treatment is cost effective only when incremental NMB is greater than zero.",
            "Population EVPI is undiscounted per-person EVPI multiplied by population size.",
            "Directional EVoP chooses by expected NMB under one perspective and evaluates regret under another.",
        ],
        "units": {
            "cost": {
                "symbol": "NZD",
                "dimension": "currency",
                "currency_code": "NZD",
                "currency_year": 2026,
            },
            "health": {"symbol": "QALY", "dimension": "health"},
            "wtp": {"symbol": "NZD/QALY", "dimension": "currency_per_health"},
            "population": {"symbol": "person", "dimension": "count"},
        },
        "cases": [
            {
                "case_id": "cea_one_state_two_cycle",
                "calculation": "discounted_markov_cost_effectiveness",
                "derivation": [
                    "With identity transition and discount rate 0, each two-cycle total is 2*per_cycle_value.",
                    "Health incremental cost = 2*(130-100) = 60 NZD.",
                    "Incremental QALY = 2*(1.01-1.00) = 0.02 QALY.",
                    "Health ICER = 60/0.02 = 3000 NZD/QALY.",
                    "Health incremental NMB = 5000*0.02-60 = 40 NZD.",
                    "Societal per-cycle costs are health + additional societal + human-capital productivity.",
                    "Societal incremental cost = 2*((130+5+10)-(100+20+30)) = -10 NZD.",
                    "Societal ICER = -10/0.02 = -500 NZD/QALY; incremental NMB = 5000*0.02-(-10) = 110 NZD.",
                ],
                "inputs": {
                    "states": ["alive"],
                    "cycles": 2,
                    "initial_population": [1.0],
                    "transition_matrices": {
                        "standard_care": [[1.0]],
                        "new_treatment": [[1.0]],
                    },
                    "discount_rate": 0.0,
                    "costs": {
                        "health_system": {
                            "standard_care": [100.0],
                            "new_treatment": [130.0],
                        },
                        "societal": {
                            "standard_care": [20.0],
                            "new_treatment": [5.0],
                        },
                    },
                    "productivity_costs": {
                        "human_capital": {
                            "standard_care": [30.0],
                            "new_treatment": [10.0],
                        }
                    },
                    "qalys": {
                        "standard_care": [1.0],
                        "new_treatment": [1.01],
                    },
                    "wtp_threshold": 5000.0,
                },
                "expected": {
                    "health_system": {
                        "cost_standard_care": 200.0,
                        "cost_new_treatment": 260.0,
                        "qalys_standard_care": 2.0,
                        "qalys_new_treatment": 2.02,
                        "incremental_cost": 60.0,
                        "incremental_qalys": 0.02,
                        "icer_status": "finite",
                        "icer_value": 3000.0,
                        "incremental_nmb": 40.0,
                        "is_cost_effective": True,
                    },
                    "societal_human_capital": {
                        "cost_standard_care": 300.0,
                        "cost_new_treatment": 290.0,
                        "qalys_standard_care": 2.0,
                        "qalys_new_treatment": 2.02,
                        "incremental_cost": -10.0,
                        "incremental_qalys": 0.02,
                        "icer_status": "finite",
                        "icer_value": -500.0,
                        "incremental_nmb": 110.0,
                        "is_cost_effective": True,
                    },
                },
            },
            {
                "case_id": "evpi_two_draw_crossing",
                "calculation": "expected_value_of_perfect_information",
                "derivation": [
                    "Standard-care NMB is [0,0] NZD and new-treatment NMB is [100,-100] NZD.",
                    "Perfect-information expected NMB = mean([100,0]) = 50 NZD.",
                    "Current-information NMB = max(mean([0,0]),mean([100,-100])) = 0 NZD.",
                    "EVPI per person = 50-0 = 50 NZD; population EVPI = 50*1000 = 50000 NZD.",
                ],
                "inputs": {
                    "wtp_threshold": 100.0,
                    "draws": [
                        {
                            "cost_sc": 0.0,
                            "qaly_sc": 0.0,
                            "cost_nt": 0.0,
                            "qaly_nt": 1.0,
                        },
                        {
                            "cost_sc": 0.0,
                            "qaly_sc": 0.0,
                            "cost_nt": 100.0,
                            "qaly_nt": 0.0,
                        },
                    ],
                    "target_population_size": 1000,
                },
                "expected": {"evpi_per_person": 50.0, "population_evpi": 50000.0},
            },
            {
                "case_id": "directional_evop_opposed_choices",
                "calculation": "directional_expected_value_of_perspective",
                "derivation": [
                    "Expected NMB under health_system selects A (10 versus 0).",
                    "Evaluating A instead of societal-optimal B under societal loses 100 NZD per draw.",
                    "The reverse direction selects B then loses 10 NZD per draw under health_system.",
                    "The draw-level optimal strategies disagree on both draws, so discordance is 1.",
                ],
                "inputs": {
                    "axis_order": ["draw", "strategy", "perspective"],
                    "strategies": ["A", "B"],
                    "perspectives": ["health_system", "societal"],
                    "net_benefit": [
                        [[10.0, 0.0], [0.0, 100.0]],
                        [[10.0, 0.0], [0.0, 100.0]],
                    ],
                    "decision_rule": "expected_value",
                    "selection_tie_policy": "first",
                },
                "expected": {
                    "health_system_to_societal": 100.0,
                    "societal_to_health_system": 10.0,
                    "discordance_probability": 1.0,
                },
            },
        ],
    }


def fixture_metadata_document() -> dict[str, object]:
    """Describe fixture validation without assuming a consumer language or binding."""
    return {
        "schema_version": "1.0.0",
        "fixture_id": "typed-pipeline-records-v1",
        "fixture_version": "1.0.0",
        "binding": "language_runtime_neutral",
        "logical_schema_id": "typed_pipeline_records",
        "logical_schema_version": CONTRACT_VERSION,
        "logical_schema_fingerprint": schema_fingerprint(TYPED_PIPELINE_ARROW_SCHEMA),
        "record_identity_fields": [
            "run_id",
            "intervention",
            "perspective",
            "productivity_cost_method",
        ],
        "record_order": "preserved",
        "null_semantics": "JSON null equals Arrow null",
        "float_comparison": {
            "rule": "abs(actual-expected) <= atol + rtol*abs(expected)",
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
        },
        "cross_format_invariants": [
            "logical field names, order, Arrow types, and nullability match the declared schema",
            "record values and record order match canonical JSON",
            "Arrow IPC and Parquet carry the declared fixture metadata digest",
            "schema fingerprints exclude container metadata",
        ],
        "formats": {
            "canonical_values": "typed-pipeline-records.json",
            "arrow_ipc": "typed-pipeline-records.arrow",
            "parquet": "typed-pipeline-records.parquet",
        },
        "provenance": {
            "source_id": "fixture:typed-pipeline-records-v1",
            "producer": "vop_poc_nz",
            "status": "synthetic_conformance_fixture",
        },
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


def _field_names(fields: Sequence[object], *, document: str) -> list[str]:
    names: list[str] = []
    for field in fields:
        if (
            not isinstance(field, dict)
            or set(field) != {"name", "arrow_type", "nullable", "unit"}
            or not isinstance(field.get("name"), str)
            or not isinstance(field.get("arrow_type"), str)
            or not isinstance(field.get("nullable"), bool)
        ):
            raise IncompatibleContractChange(f"{document} schema fields are invalid")
        unit = field.get("unit")
        if unit is not None and (
            not isinstance(unit, dict)
            or set(unit) != {"symbol_field", "dimension"}
            or (
                unit.get("symbol_field") is not None
                and not isinstance(unit.get("symbol_field"), str)
            )
            or not isinstance(unit.get("dimension"), str)
        ):
            raise IncompatibleContractChange(f"{document} schema field unit is invalid")
        name = cast(str, field["name"])
        names.append(name)
    if has_name_collision(names):
        raise IncompatibleContractChange(f"{document} schema field names collide")
    return names


def _added_field_names(
    additions: Sequence[object], *, existing_names: set[str]
) -> list[str]:
    names: list[str] = []
    for addition in additions:
        if not isinstance(addition, dict):
            raise IncompatibleContractChange("new fields must be nullable")
        name = addition.get("name")
        if addition.get("nullable") is not True or not isinstance(name, str):
            raise IncompatibleContractChange("new fields must be nullable")
        if has_name_collision((*names, name), existing_names):
            raise IncompatibleContractChange("new schema field names collide")
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
            "vop_voiage.fixture_binding",
            "vop_voiage.fixture_id",
            "vop_voiage.fixture_metadata_sha256",
            "vop_voiage.fixture_version",
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


def _validate_computed_fingerprints(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    previous_fields: Sequence[Mapping[str, object]],
    current_fields: Sequence[Mapping[str, object]],
) -> None:
    """Require migration descriptors to match represented Arrow fields."""
    arrow_types: dict[str, pa.DataType] = {
        "bool": pa.bool_(),
        "double": pa.float64(),
        "int64": pa.int64(),
        "string": pa.string(),
    }
    computed: list[str] = []
    for document, fields in (
        ("previous", previous_fields),
        ("current", current_fields),
    ):
        try:
            schema = pa.schema(
                [
                    pa.field(
                        str(field["name"]),
                        arrow_types[str(field["arrow_type"])],
                        nullable=bool(field["nullable"]),
                    )
                    for field in fields
                    if isinstance(field, Mapping)
                ]
            )
        except (KeyError, TypeError) as exc:
            raise IncompatibleContractChange(
                f"unsupported {document} Arrow field identity"
            ) from exc
        computed.append(schema_fingerprint(schema))
    if not matches_computed_identity(previous.get("schema_fingerprint"), computed[0]):
        raise IncompatibleContractChange("previous schema fingerprint mismatch")
    if not matches_computed_identity(current.get("schema_fingerprint"), computed[1]):
        raise IncompatibleContractChange("current schema fingerprint mismatch")


def assess_arrow_evolution(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> dict[str, object]:
    """Assess an identity transition, rejecting all undeclared incompatibility."""
    if not has_exact_keys(previous, _ARROW_IDENTITY_KEYS) or not has_exact_keys(
        current, _ARROW_IDENTITY_KEYS
    ):
        raise IncompatibleContractChange("unknown top-level schema semantics")
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
    previous_names = _field_names(previous_fields, document="previous")
    _field_names(current_fields, document="current")
    previous_descriptors = cast(list[Mapping[str, object]], previous_fields)
    current_descriptors = cast(list[Mapping[str, object]], current_fields)
    if len(current_fields) < len(previous_fields):
        raise IncompatibleContractChange("schema fields were removed")
    if current_fields[: len(previous_fields)] != previous_fields:
        raise IncompatibleContractChange(
            "existing dtype, unit, nullability, or order changed"
        )
    additions = current_fields[len(previous_fields) :]
    added_fields = _added_field_names(additions, existing_names=set(previous_names))
    _validate_fingerprint_transition(
        previous.get("schema_fingerprint"),
        current.get("schema_fingerprint"),
        changed=bool(additions),
    )
    _validate_computed_fingerprints(
        previous, current, previous_descriptors, current_descriptors
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
    fixture = output / "fixtures"
    fixture.mkdir(parents=True, exist_ok=True)
    fixture_metadata = fixture_metadata_document()
    fixture_metadata_bytes = canonical_json_bytes(fixture_metadata)
    fixture_metadata_digest = sha256(fixture_metadata_bytes).hexdigest()
    (fixture / "typed-pipeline-records.metadata.json").write_bytes(
        fixture_metadata_bytes
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
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"vop_voiage.fixture_binding": b"language_runtime_neutral",
            b"vop_voiage.fixture_id": b"typed-pipeline-records-v1",
            b"vop_voiage.fixture_version": b"1.0.0",
            b"vop_voiage.fixture_metadata_sha256": fixture_metadata_digest.encode(),
        }
    )
    (fixture / "typed-pipeline-records.json").write_bytes(
        canonical_json_bytes(
            {
                "fixture_binding": "language_runtime_neutral",
                "fixture_id": "typed-pipeline-records-v1",
                "fixture_metadata": "typed-pipeline-records.metadata.json",
                "fixture_metadata_sha256": fixture_metadata_digest,
                "fixture_version": "1.0.0",
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


def _write_analytical_reference(output: Path) -> tuple[str, str]:
    reference = output / "references" / "analytical-reference-manifest.json"
    reference.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(analytical_reference_document())
    reference.write_bytes(content)
    return reference.relative_to(output).as_posix(), sha256(content).hexdigest()


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
    reference_path, reference_digest = _write_analytical_reference(output)
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
        "analytical_reference": {
            "path": reference_path,
            "reference_id": "vop-voiage-analytical-reference",
            "reference_version": "1.0.0",
            "sha256": reference_digest,
        },
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
    reference = raw.get("analytical_reference")
    reference_path = "references/analytical-reference-manifest.json"
    if not isinstance(reference, dict) or reference != {
        "path": reference_path,
        "reference_id": "vop-voiage-analytical-reference",
        "reference_version": "1.0.0",
        "sha256": sha256((output / reference_path).read_bytes()).hexdigest(),
    }:
        raise ValueError("contract bundle analytical reference identity mismatch")
    return raw


__all__ = [
    "BUNDLE_ID",
    "BUNDLE_VERSION",
    "IncompatibleContractChange",
    "analytical_reference_document",
    "arrow_identity_document",
    "assess_arrow_evolution",
    "canonical_json_bytes",
    "export_contract_bundle",
    "fixture_metadata_document",
    "migration_policy_document",
    "verify_contract_bundle",
]
