"""Arrow-first input/output helpers for perspective-regret outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .perspective import NetBenefitTensor

ARROW_SCHEMA_VERSION = "1.0.0"


def schema_fingerprint(schema: pa.Schema) -> str:
    """Return a stable SHA-256 fingerprint for an Arrow logical schema.

    Container-specific metadata is excluded so the same logical table has the
    same cross-language identity in Parquet, Arrow IPC, PyArrow, and Polars.
    """
    fields = [
        {"arrow_type": str(field.type), "name": field.name, "nullable": field.nullable}
        for field in schema.remove_metadata()
    ]
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def records_to_arrow(records: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Build a schema-bearing Arrow table from row mappings."""
    table = pa.Table.from_pylist([dict(row) for row in records])
    fingerprint = schema_fingerprint(table.schema)
    return table.replace_schema_metadata(
        {
            b"vop.arrow_schema_version": ARROW_SCHEMA_VERSION.encode(),
            b"vop.schema_fingerprint": fingerprint.encode(),
            b"vop_voiage.contract_version": b"1.0.0",
            b"vop_voiage.schema_id": b"net_benefit_records",
            b"vop_voiage.schema_version": ARROW_SCHEMA_VERSION.encode(),
            b"vop_voiage.schema_fingerprint": fingerprint.encode(),
            b"vop_voiage.producer": b"vop_poc_nz",
            b"vop_voiage.method_contract_version": b"1.1.0",
        }
    )


def tensor_to_long_records(tensor: NetBenefitTensor) -> list[dict[str, Any]]:
    """Convert a `NetBenefitTensor` to long-form records."""
    records: list[dict[str, Any]] = []
    draw_ids = tensor.draw_ids or tuple(str(i) for i in range(tensor.n_draws))
    for draw_idx, draw_id in enumerate(draw_ids):
        for strategy_idx, strategy in enumerate(tensor.strategies):
            for perspective_idx, perspective in enumerate(tensor.perspectives):
                records.append(
                    {
                        "case_id": tensor.case_id,
                        "draw": draw_id,
                        "strategy": strategy,
                        "perspective": perspective,
                        "net_benefit": float(
                            tensor.values[draw_idx, strategy_idx, perspective_idx]
                        ),
                    }
                )
    return records


def write_records(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    prefer_parquet: bool = True,
) -> Path:
    """Write records to schema-bearing Parquet or explicit JSON Lines.

    Args:
        records: Iterable of JSON-like row dictionaries.
        path: Output path. If the suffix is `.parquet`, Parquet is attempted.
        prefer_parquet: If true, paths without `.jsonl` use Parquet. Arrow is a
            required dependency; serialization errors are never hidden by an
            implicit format fallback.
    """
    rows = [dict(row) for row in records]
    out = Path(path)
    if prefer_parquet and out.suffix not in {".parquet", ".jsonl"}:
        out = out.with_suffix(".parquet")

    if (prefer_parquet and out.suffix != ".jsonl") or out.suffix == ".parquet":
        if out.suffix != ".parquet":
            out = out.with_suffix(".parquet")
        table = records_to_arrow(rows)
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                b"vop.interchange": b"apache-arrow-parquet",
                b"vop_voiage.interchange": b"apache-arrow-parquet",
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out, compression="zstd", version="2.6")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def read_records(path: str | Path) -> pa.Table:
    """Read a Parquet artifact as an Arrow table without pandas conversion."""
    source = Path(path)
    if source.suffix != ".parquet":
        raise ValueError("Arrow interchange requires a .parquet artifact")
    return pq.read_table(source)


def write_ipc_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write records as an Arrow IPC file with the shared schema identity."""
    target = Path(path).with_suffix(".arrow")
    table = records_to_arrow(records)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"vop.interchange": b"apache-arrow-ipc",
            b"vop_voiage.interchange": b"apache-arrow-ipc",
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_file(target, table.schema) as writer:
        writer.write_table(table)
    return target


def read_ipc_records(path: str | Path) -> pa.Table:
    """Read an Arrow IPC file without a pandas conversion."""
    source = Path(path)
    if source.suffix != ".arrow":
        raise ValueError("Arrow IPC interchange requires a .arrow artifact")
    with ipc.open_file(source) as reader:
        return reader.read_all()


def to_arrow_c_stream(records: Iterable[Mapping[str, Any]]) -> object:
    """Expose records through Arrow's zero-copy PyCapsule stream protocol."""
    table = records_to_arrow(records)
    return table.__arrow_c_stream__()


def write_perspective_outputs(
    tensor: NetBenefitTensor,
    output_dir: str | Path,
    *,
    reference_perspective: str,
    population: float | None = None,
) -> dict[str, Path]:
    """Write standard perspective-analysis outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "net_benefit_tensor": write_records(
            tensor_to_long_records(tensor), out / "net_benefit_tensor.parquet"
        ),
        "regret_matrix": write_records(
            tensor.regret_matrix(population=population), out / "regret_matrix.parquet"
        ),
        "perspective_acceptability_frontier": write_records(
            (row.as_dict() for row in tensor.perspective_acceptability_frontier()),
            out / "perspective_acceptability_frontier.parquet",
        ),
        "mcda_features": write_records(
            tensor.mcda_feature_records(
                reference_perspective=reference_perspective, population=population
            ),
            out / "mcda_features.parquet",
        ),
    }
