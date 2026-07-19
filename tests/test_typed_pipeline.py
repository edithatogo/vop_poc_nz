from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vop_poc_nz.cea_model_core import run_cea
from vop_poc_nz.logging_config import LoggingSettings, configure_logging
from vop_poc_nz.perspective_io import read_records, write_records
from vop_poc_nz.pipeline.typed import (
    pipeline_result_records,
    run_typed_analysis_pipeline,
    typed_pipeline_spec_from_legacy,
)
from vop_poc_nz.result_manifest import build_result_manifest


def _parameters(cost: float = 15.0) -> dict[str, object]:
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
            "new_treatment": [1.0, 0.0],
        },
        "discount_rate": 0.03,
        "productivity_costs": {
            "human_capital": {
                "standard_care": [4.0, 0.0],
                "new_treatment": [1.0, 0.0],
            }
        },
        "friction_cost_params": {
            "friction_period_days": 90,
            "replacement_cost_per_day": 100.0,
            "absenteeism_rate": 0.05,
        },
        "productivity_loss_states": {"Healthy": 2.0},
    }


def _pipeline_spec():
    return typed_pipeline_spec_from_legacy(
        {"A": _parameters(), "B": _parameters(20.0)},
        run_id="run-123",
        created_at_utc=datetime(2026, 1, 2, tzinfo=UTC),
        random_seed=42,
        software_version="0.2.2",
    )


def test_pipeline_spec_and_result_are_immutable_and_do_not_mutate_input() -> None:
    source = {"A": _parameters()}
    before = copy.deepcopy(source)
    spec = typed_pipeline_spec_from_legacy(
        source,
        run_id="run-immutable",
        created_at_utc=datetime(2026, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        spec.run_id = "changed"  # type: ignore[misc]

    result = run_typed_analysis_pipeline(spec)

    assert source == before
    assert result.spec_fingerprint == spec.spec_fingerprint
    with pytest.raises(ValidationError):
        result.run_id = "changed"  # type: ignore[misc]
    assert source == before


def test_legacy_adapter_has_exact_cea_numeric_parity() -> None:
    parameters = _parameters()
    result = run_typed_analysis_pipeline(
        typed_pipeline_spec_from_legacy(
            {"A": parameters},
            run_id="run-parity",
            created_at_utc=datetime(2026, 1, 2, tzinfo=UTC),
        )
    )
    legacy = result.to_legacy_intervention_results()["A"]

    expected_health = run_cea(parameters, perspective="health_system")
    expected_societal = run_cea(
        parameters,
        perspective="societal",
        productivity_cost_method="human_capital",
    )
    for key in ("incremental_cost", "incremental_qalys", "incremental_nmb", "icer"):
        assert legacy["health_system"][key] == pytest.approx(expected_health[key])
        assert legacy["societal"]["human_capital"][key] == pytest.approx(
            expected_societal[key]
        )


def test_pipeline_emits_structured_context_and_provenance(capsys) -> None:
    configure_logging(
        LoggingSettings(
            json_output=True,
            console=True,
            run_id="logger-run",
            level="INFO",
        )
    )
    result = run_typed_analysis_pipeline(_pipeline_spec())
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    calculations = [
        event
        for event in events
        if event["message"] == "typed_cea_calculation_completed"
    ]

    assert {event["intervention"] for event in calculations} == {"A", "B"}
    assert {event["pipeline_run_id"] for event in calculations} == {"run-123"}
    assert all(
        event["spec_fingerprint"] == result.spec_fingerprint for event in calculations
    )
    assert result.random_seed == 42
    assert result.software_version == "0.2.2"


def test_pipeline_records_support_arrow_and_result_manifest(tmp_path) -> None:
    result = run_typed_analysis_pipeline(_pipeline_spec())
    records = pipeline_result_records(result)
    artifact = write_records(records, tmp_path / "typed-pipeline.parquet")
    table = read_records(artifact)

    assert table.num_rows == 6
    assert set(table.column("perspective").to_pylist()) == {"health_system", "societal"}
    manifest = build_result_manifest(
        run_id=result.run_id,
        script="vop_poc_nz.pipeline.typed",
        outputs=[artifact],
        random_seed=result.random_seed,
        parameters={"spec_fingerprint": result.spec_fingerprint},
    )
    assert manifest.outputs[0].sha256
    assert manifest.parameters["spec_fingerprint"] == result.spec_fingerprint
