from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from vop_poc_nz.cea_model_core import run_cea
from vop_poc_nz.compat.legacy import intervention_spec_from_legacy, run_typed_cea
from vop_poc_nz.domain.cea import NumericVector
from vop_poc_nz.kernels.base import CalculationKernel
from vop_poc_nz.kernels.cea import CEACalculationKernel
from vop_poc_nz.results.base import AnalysisResult
from vop_poc_nz.results.cea import ICERResult, ICERStatus


def _parameters() -> dict[str, object]:
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
                "new_treatment": [15.0, 0.0],
            },
            "societal": {
                "standard_care": [2.0, 0.0],
                "new_treatment": [-1.0, 0.0],
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
        "subgroups": {
            "priority": {
                "initial_population": [40.0, 0.0],
                "transition_matrices": {"new_treatment": [[0.95, 0.05], [0.0, 1.0]]},
            },
            "other": {"initial_population": [60.0, 0.0]},
        },
    }


def test_domain_is_strict_and_deeply_immutable() -> None:
    with pytest.raises(ValidationError):
        NumericVector(values=[1.0, 2.0])  # type: ignore[arg-type]

    spec = intervention_spec_from_legacy(_parameters())
    with pytest.raises(ValidationError):
        spec.cycles = 4  # type: ignore[misc]
    assert isinstance(spec.states, tuple)
    assert isinstance(spec.transition_matrices.standard_care.rows[0], tuple)
    assert isinstance(spec.subgroups, tuple)


def test_domain_validates_dimensions_and_stochastic_rows() -> None:
    malformed = _parameters()
    malformed["transition_matrices"] = {
        "standard_care": [[0.7, 0.2], [0.0, 1.0]],
        "new_treatment": [[0.9, 0.1], [0.0, 1.0]],
    }
    with pytest.raises(ValidationError, match="sum to 1"):
        intervention_spec_from_legacy(malformed)


def test_domain_validates_subgroup_override_dimensions() -> None:
    malformed = _parameters()
    malformed["subgroups"] = {"priority": {"qalys": {"new_treatment": [1.0]}}}
    with pytest.raises(ValidationError, match=r"subgroup 'priority'.*state count"):
        intervention_spec_from_legacy(malformed)


@pytest.mark.parametrize(
    ("value", "status", "typed_value"),
    [
        (120.0, ICERStatus.FINITE, 120.0),
        (math.inf, ICERStatus.DOMINATED, None),
        (-math.inf, ICERStatus.DOMINANT, None),
        (math.nan, ICERStatus.UNDEFINED, None),
    ],
)
def test_icer_result_uses_typed_status(
    value: float, status: ICERStatus, typed_value: float | None
) -> None:
    result = ICERResult.from_legacy(value)
    assert result.status is status
    assert result.value == typed_value


@pytest.mark.parametrize(
    ("perspective", "productivity_method"),
    [("health_system", "human_capital"), ("societal", "friction_cost")],
)
def test_typed_kernel_has_legacy_numeric_parity(
    perspective: str, productivity_method: str
) -> None:
    parameters = _parameters()
    legacy = run_cea(
        parameters,
        perspective=perspective,
        productivity_cost_method=productivity_method,
    )
    typed = run_typed_cea(
        parameters,
        perspective=perspective,
        productivity_cost_method=productivity_method,
    )
    restored = typed.to_legacy_dict()

    for key in (
        "cost_standard_care",
        "qalys_standard_care",
        "cost_new_treatment",
        "qalys_new_treatment",
        "incremental_cost",
        "incremental_qalys",
        "incremental_nmb",
    ):
        assert restored[key] == pytest.approx(legacy[key])
    assert restored["icer"] == legacy["icer"]
    assert np.array_equal(
        restored["trace_standard_care"], legacy["trace_standard_care"]
    )
    assert np.array_equal(
        restored["trace_new_treatment"], legacy["trace_new_treatment"]
    )
    assert restored["subgroup_results"].keys() == legacy["subgroup_results"].keys()


def test_kernel_contract_and_result_are_immutable() -> None:
    kernel = CEACalculationKernel()
    assert isinstance(kernel, CalculationKernel)
    result = run_typed_cea(_parameters())
    assert isinstance(result, AnalysisResult)
    with pytest.raises(ValidationError):
        result.incremental_cost = 0.0  # type: ignore[misc]
    assert isinstance(result.trace_standard_care, tuple)
    assert isinstance(result.trace_standard_care[0], tuple)
