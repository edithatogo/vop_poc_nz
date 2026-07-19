"""Arrow-first perspective interchange tests."""

import pyarrow as pa

from vop_poc_nz.perspective_io import read_records, to_arrow_c_stream, write_records


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
