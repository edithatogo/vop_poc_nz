"""Production checks against literal, hand-derived analytical references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from vop_poc_nz.compat.legacy import intervention_spec_from_legacy, run_typed_cea
from vop_poc_nz.domain.contracts import MetadataStatus, UnitDimension, UnitSpec
from vop_poc_nz.perspective import NetBenefitTensor
from vop_poc_nz.results.cea import CEAAnalysisResult
from vop_poc_nz.value_of_information import calculate_evpi, calculate_population_evpi

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "contracts/vop-voiage/1.0.0/references/analytical-reference-manifest.json"
)


def _document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(REFERENCE.read_text(encoding="utf-8")))


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in _document()["cases"] if case["case_id"] == case_id)


def _assert_cea(actual: CEAAnalysisResult, expected: dict[str, Any]) -> None:
    for field in (
        "cost_standard_care",
        "cost_new_treatment",
        "qalys_standard_care",
        "qalys_new_treatment",
        "incremental_cost",
        "incremental_qalys",
        "incremental_nmb",
    ):
        assert getattr(actual, field) == pytest.approx(
            expected[field], abs=1e-12, rel=1e-12
        )
    assert actual.icer.status.value == expected["icer_status"]
    assert actual.icer.value == pytest.approx(
        expected["icer_value"], abs=1e-12, rel=1e-12
    )
    assert actual.is_cost_effective is expected["is_cost_effective"]


def test_reference_is_explicitly_independent_binding_neutral_and_versioned() -> None:
    document = _document()
    assert document["schema_version"] == "1.0.0"
    assert document["reference_version"] == "1.0.0"
    assert document["binding"] == "language_runtime_neutral"
    assert document["provenance"] == {
        "derivation_date": "2026-07-20",
        "derivation_method": "hand_calculated_closed_form",
        "implementation_dependency": "none",
        "review_status": "producer_authored_pending_external_replication",
        "source_id": "manual-algebra:vop-voiage-reference-v1",
    }
    assert len(document["assumptions"]) >= 5
    assert set(document["units"]) == {"cost", "health", "population", "wtp"}
    assert {case["case_id"] for case in document["cases"]} == {
        "cea_one_state_two_cycle",
        "evpi_two_draw_crossing",
        "directional_evop_opposed_choices",
    }
    assert all(case["derivation"] for case in document["cases"])


def test_hand_derived_cea_case_matches_typed_production_kernel() -> None:
    case = _case("cea_one_state_two_cycle")
    inputs = cast(dict[str, Any], case["inputs"])
    parameters = {key: value for key, value in inputs.items() if key != "wtp_threshold"}
    spec = intervention_spec_from_legacy(parameters).model_copy(
        update={
            "cost_unit": UnitSpec(
                symbol="NZD",
                dimension=UnitDimension.CURRENCY,
                currency_code="NZD",
                currency_year=2026,
                metadata_status=MetadataStatus.KNOWN,
            )
        }
    )
    expected = cast(dict[str, dict[str, Any]], case["expected"])
    health = run_typed_cea(
        spec,
        perspective="health_system",
        wtp_threshold=inputs["wtp_threshold"],
    )
    societal = run_typed_cea(
        spec,
        perspective="societal",
        productivity_cost_method="human_capital",
        wtp_threshold=inputs["wtp_threshold"],
    )
    _assert_cea(health, expected["health_system"])
    _assert_cea(societal, expected["societal_human_capital"])
    assert health.cost_unit.currency_code == societal.cost_unit.currency_code == "NZD"
    assert health.cost_unit.currency_year == societal.cost_unit.currency_year == 2026
    assert health.health_outcome_unit.symbol == "QALY"


def test_hand_derived_evpi_and_population_case_match_production() -> None:
    case = _case("evpi_two_draw_crossing")
    inputs = cast(dict[str, Any], case["inputs"])
    expected = cast(dict[str, float], case["expected"])
    evpi = calculate_evpi(
        pd.DataFrame(inputs["draws"]), wtp_threshold=inputs["wtp_threshold"]
    )
    population = calculate_population_evpi(evpi, inputs["target_population_size"])
    assert evpi == pytest.approx(expected["evpi_per_person"], abs=1e-12, rel=1e-12)
    assert population == pytest.approx(
        expected["population_evpi"], abs=1e-12, rel=1e-12
    )


def test_hand_derived_directional_evop_case_matches_production() -> None:
    case = _case("directional_evop_opposed_choices")
    inputs = cast(dict[str, Any], case["inputs"])
    expected = cast(dict[str, float], case["expected"])
    tensor = NetBenefitTensor(
        np.asarray(inputs["net_benefit"], dtype=np.float64),
        strategies=tuple(inputs["strategies"]),
        perspectives=tuple(inputs["perspectives"]),
    )
    forward = tensor.evop(
        choose_under="health_system",
        evaluate_under="societal",
        decision_rule=inputs["decision_rule"],
        selection_tie_policy=inputs["selection_tie_policy"],
    )
    reverse = tensor.evop(
        choose_under="societal",
        evaluate_under="health_system",
        decision_rule=inputs["decision_rule"],
        selection_tie_policy=inputs["selection_tie_policy"],
    )
    assert forward.per_person == pytest.approx(
        expected["health_system_to_societal"], abs=1e-12, rel=1e-12
    )
    assert reverse.per_person == pytest.approx(
        expected["societal_to_health_system"], abs=1e-12, rel=1e-12
    )
    assert tensor.discordance_probability("health_system", "societal") == pytest.approx(
        expected["discordance_probability"], abs=1e-12, rel=1e-12
    )
