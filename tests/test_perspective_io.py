"""Arrow-first perspective interchange tests."""

import json
from pathlib import Path

import polars as pl
import pyarrow as pa

from vop_poc_nz.perspective_io import (
    read_ipc_records,
    read_records,
    schema_fingerprint,
    to_arrow_c_stream,
    write_ipc_records,
    write_records,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "interchange" / "v1"


def test_parquet_round_trip_is_schema_bearing(tmp_path) -> None:
    path = write_records(
        [{"draw": "0", "strategy": "A", "perspective": "societal", "net_benefit": 4.0}],
        tmp_path / "tensor.parquet",
    )
    table = read_records(path)

    assert isinstance(table, pa.Table)
    assert table.to_pylist()[0]["net_benefit"] == 4.0
    assert table.schema.metadata[b"vop.arrow_schema_version"] == b"1.0.0"


def test_arrow_c_stream_capsule_is_available() -> None:
    capsule = to_arrow_c_stream([{"value": 1}])
    assert type(capsule).__name__ == "PyCapsule"


def test_pyarrow_polars_round_trip_preserves_nulls_and_schema(tmp_path: Path) -> None:
    rows = [
        {"category": "a", "value": 1.5, "nullable": None},
        {"category": "b", "value": -2.0, "nullable": "present"},
    ]
    parquet_path = write_records(rows, tmp_path / "roundtrip.parquet")
    arrow_table = read_records(parquet_path)
    frame = pl.from_arrow(arrow_table)
    assert isinstance(frame, pl.DataFrame)

    ipc_path = write_ipc_records(
        frame.to_arrow().to_pylist(), tmp_path / "roundtrip.arrow"
    )
    restored = read_ipc_records(ipc_path)

    assert restored.to_pylist() == rows
    assert schema_fingerprint(restored.schema) == schema_fingerprint(arrow_table.schema)


def test_versioned_golden_parquet_and_ipc_contract() -> None:
    contract = json.loads((FIXTURE_ROOT / "contract.json").read_text(encoding="utf-8"))
    parquet = read_records(FIXTURE_ROOT / "perspective.parquet")
    arrow = read_ipc_records(FIXTURE_ROOT / "perspective.arrow")

    assert parquet.to_pylist() == contract["records"]
    assert arrow.to_pylist() == contract["records"]
    assert schema_fingerprint(parquet.schema) == contract["schema_fingerprint"]
    assert schema_fingerprint(arrow.schema) == contract["schema_fingerprint"]
    assert parquet.schema.metadata[b"vop.arrow_schema_version"] == b"1.0.0"
