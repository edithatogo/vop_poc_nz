"""Hypothesis and metamorphic assurance across typed and interchange boundaries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vop_poc_nz.compat.legacy import intervention_spec_from_legacy, run_typed_cea
from vop_poc_nz.perspective_io import attach_contract_metadata, schema_fingerprint


def _parameters(cost: float, discount_rate: float) -> dict[str, object]:
    return {
        "states": ["Healthy", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.8, 0.2], [0.0, 1.0]],
            "new_treatment": [[0.9, 0.1], [0.0, 1.0]],
        },
        "cycles": 3,
        "initial_population": [100.0, 0.0],
        "costs": {
            "health_system": {
                "standard_care": [10.0, 0.0],
                "new_treatment": [cost, 0.0],
            },
            "societal": {
                "standard_care": [2.0, 0.0],
                "new_treatment": [1.0, 0.0],
            },
        },
        "qalys": {
            "standard_care": [1.0, 0.0],
            "new_treatment": [1.1, 0.0],
        },
        "discount_rate": discount_rate,
    }


@settings(max_examples=25, deadline=None)
@given(
    cost=st.floats(
        min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False
    ),
    discount_rate=st.floats(
        min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False
    ),
)
def test_typed_legacy_round_trip_has_exact_numerical_parity(
    cost: float, discount_rate: float
) -> None:
    source = _parameters(cost, discount_rate)
    typed = intervention_spec_from_legacy(source)
    typed_from_json = type(typed).model_validate_json(typed.model_dump_json())
    result_from_mapping = run_typed_cea(source)
    result_from_typed = run_typed_cea(typed)
    assert typed_from_json == typed
    assert result_from_typed == result_from_mapping


@settings(max_examples=20, deadline=None)
@given(
    values=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=64),
        min_size=1,
        max_size=20,
    )
)
def test_json_arrow_ipc_parquet_round_trip_is_metamorphic(
    values: list[float],
) -> None:
    records = [{"index": index, "value": value} for index, value in enumerate(values)]
    json_records = json.loads(json.dumps(records, sort_keys=True))
    table = attach_contract_metadata(
        pa.Table.from_pylist(records),
        schema_id="hypothesis_numeric_records",
        provenance_json=json.dumps(
            [
                {
                    "source_id": "hypothesis",
                    "observed_at_utc": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                }
            ],
            sort_keys=True,
        ),
    )
    arrow_sink = pa.BufferOutputStream()
    with ipc.new_file(arrow_sink, table.schema) as writer:
        writer.write_table(table)
    parquet_sink = pa.BufferOutputStream()
    pq.write_table(table, parquet_sink)
    with ipc.open_file(arrow_sink.getvalue()) as reader:
        arrow = reader.read_all()
    parquet = pq.read_table(parquet_sink.getvalue())
    assert arrow.to_pylist() == parquet.to_pylist() == json_records
    assert arrow.schema.metadata == parquet.schema.metadata == table.schema.metadata
    assert schema_fingerprint(arrow.schema) == schema_fingerprint(parquet.schema)


@settings(max_examples=20, deadline=None)
@given(
    scale=st.floats(
        min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
    )
)
def test_cost_scale_metamorphism_preserves_qalys_and_scales_costs(scale: float) -> None:
    base = run_typed_cea(_parameters(20.0, 0.03))
    scaled = _parameters(20.0 * scale, 0.03)
    costs = cast(dict[str, object], scaled["costs"])
    health = cast(dict[str, object], costs["health_system"])
    health["standard_care"] = [10.0 * scale, 0.0]
    result = run_typed_cea(scaled)
    assert result.incremental_qalys == pytest.approx(base.incremental_qalys)
    assert result.incremental_cost == pytest.approx(base.incremental_cost * scale)
