"""Adapters between mutable legacy mappings and typed CEA contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vop_poc_nz.domain.cea import (
    ArmVectors,
    CostSpec,
    FrictionCostSpec,
    InterventionSpec,
    NamedSubgroupSpec,
    NumericVector,
    PartialArmVectors,
    PartialCostSpec,
    PartialTransitionMatrices,
    Perspective,
    ProductivityCostMethod,
    ProductivityCostSpec,
    ProductivityLossState,
    SubgroupOverrideSpec,
    TransitionMatrices,
    TransitionMatrix,
)
from vop_poc_nz.kernels.cea import CEACalculationContext, CEACalculationKernel
from vop_poc_nz.results.cea import CEAAnalysisResult


def _vector(value: Sequence[Any]) -> NumericVector:
    return NumericVector(values=tuple(float(item) for item in value))


def _matrix(value: Sequence[Sequence[Any]]) -> TransitionMatrix:
    return TransitionMatrix(
        rows=tuple(tuple(float(item) for item in row) for row in value)
    )


def _arm_vectors(value: Mapping[str, Any]) -> ArmVectors:
    return ArmVectors(
        standard_care=_vector(value["standard_care"]),
        new_treatment=_vector(value["new_treatment"]),
    )


def _partial_arm_vectors(value: Mapping[str, Any]) -> PartialArmVectors:
    return PartialArmVectors(
        standard_care=_vector(value["standard_care"])
        if "standard_care" in value
        else None,
        new_treatment=_vector(value["new_treatment"])
        if "new_treatment" in value
        else None,
    )


def _productivity_costs(value: Mapping[str, Any]) -> ProductivityCostSpec:
    return ProductivityCostSpec(human_capital=_arm_vectors(value["human_capital"]))


def _friction_costs(value: Mapping[str, Any]) -> FrictionCostSpec:
    return FrictionCostSpec(
        friction_period_days=float(value["friction_period_days"]),
        replacement_cost_per_day=float(value["replacement_cost_per_day"]),
        absenteeism_rate=float(value["absenteeism_rate"]),
    )


def _loss_states(value: Mapping[str, Any]) -> tuple[ProductivityLossState, ...]:
    return tuple(
        ProductivityLossState(state=str(state), annual_absence_days=float(days))
        for state, days in value.items()
    )


def _subgroup(name: str, value: Mapping[str, Any]) -> NamedSubgroupSpec:
    transitions = value.get("transition_matrices")
    costs = value.get("costs")
    return NamedSubgroupSpec(
        name=name,
        override=SubgroupOverrideSpec(
            initial_population=_vector(value["initial_population"])
            if "initial_population" in value
            else None,
            transition_matrices=PartialTransitionMatrices(
                standard_care=_matrix(transitions["standard_care"])
                if transitions and "standard_care" in transitions
                else None,
                new_treatment=_matrix(transitions["new_treatment"])
                if transitions and "new_treatment" in transitions
                else None,
            )
            if transitions
            else None,
            costs=PartialCostSpec(
                health_system=_partial_arm_vectors(costs["health_system"])
                if costs and "health_system" in costs
                else None,
                societal=_partial_arm_vectors(costs["societal"])
                if costs and "societal" in costs
                else None,
            )
            if costs
            else None,
            qalys=_partial_arm_vectors(value["qalys"]) if "qalys" in value else None,
            discount_rate=float(value["discount_rate"])
            if "discount_rate" in value
            else None,
            productivity_costs=_productivity_costs(value["productivity_costs"])
            if "productivity_costs" in value
            else None,
            friction_cost_params=_friction_costs(value["friction_cost_params"])
            if "friction_cost_params" in value
            else None,
            productivity_loss_states=_loss_states(value["productivity_loss_states"])
            if "productivity_loss_states" in value
            else None,
        ),
    )


def intervention_spec_from_legacy(data: Mapping[str, Any]) -> InterventionSpec:
    """Validate and deep-freeze the calculation-relevant legacy fields."""
    transitions = data["transition_matrices"]
    costs = data["costs"]
    return InterventionSpec(
        states=tuple(str(state) for state in data["states"]),
        cycles=int(data["cycles"]),
        initial_population=_vector(data["initial_population"]),
        transition_matrices=TransitionMatrices(
            standard_care=_matrix(transitions["standard_care"]),
            new_treatment=_matrix(transitions["new_treatment"]),
        ),
        costs=CostSpec(
            health_system=_arm_vectors(costs["health_system"]),
            societal=_arm_vectors(costs["societal"]),
        ),
        qalys=_arm_vectors(data["qalys"]),
        discount_rate=float(data.get("discount_rate", 0.03)),
        productivity_costs=_productivity_costs(data["productivity_costs"])
        if "productivity_costs" in data
        else None,
        friction_cost_params=_friction_costs(data["friction_cost_params"])
        if "friction_cost_params" in data
        else None,
        productivity_loss_states=_loss_states(data.get("productivity_loss_states", {})),
        subgroups=tuple(
            _subgroup(str(name), override)
            for name, override in data.get("subgroups", {}).items()
        ),
    )


def _vector_list(value: NumericVector) -> list[float]:
    return list(value.values)


def _matrix_list(value: TransitionMatrix) -> list[list[float]]:
    return [list(row) for row in value.rows]


def _arm_mapping(value: ArmVectors) -> dict[str, list[float]]:
    return {
        "standard_care": _vector_list(value.standard_care),
        "new_treatment": _vector_list(value.new_treatment),
    }


def _partial_arm_mapping(value: PartialArmVectors) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    if value.standard_care is not None:
        output["standard_care"] = _vector_list(value.standard_care)
    if value.new_treatment is not None:
        output["new_treatment"] = _vector_list(value.new_treatment)
    return output


def _partial_transition_mapping(
    value: PartialTransitionMatrices,
) -> dict[str, list[list[float]]]:
    matrices: dict[str, list[list[float]]] = {}
    if value.standard_care is not None:
        matrices["standard_care"] = _matrix_list(value.standard_care)
    if value.new_treatment is not None:
        matrices["new_treatment"] = _matrix_list(value.new_treatment)
    return matrices


def _partial_cost_mapping(value: PartialCostSpec) -> dict[str, object]:
    costs: dict[str, object] = {}
    if value.health_system is not None:
        costs["health_system"] = _partial_arm_mapping(value.health_system)
    if value.societal is not None:
        costs["societal"] = _partial_arm_mapping(value.societal)
    return costs


def _subgroup_mapping(value: SubgroupOverrideSpec) -> dict[str, object]:
    output: dict[str, object] = (
        {"initial_population": _vector_list(value.initial_population)}
        if value.initial_population is not None
        else {}
    )
    if value.transition_matrices is not None:
        output["transition_matrices"] = _partial_transition_mapping(
            value.transition_matrices
        )
    if value.costs is not None:
        output["costs"] = _partial_cost_mapping(value.costs)
    if value.qalys is not None:
        output["qalys"] = _partial_arm_mapping(value.qalys)
    if value.discount_rate is not None:
        output["discount_rate"] = value.discount_rate
    if value.productivity_costs is not None:
        output["productivity_costs"] = {
            "human_capital": _arm_mapping(value.productivity_costs.human_capital)
        }
    if value.friction_cost_params is not None:
        output["friction_cost_params"] = value.friction_cost_params.model_dump()
    if value.productivity_loss_states is not None:
        output["productivity_loss_states"] = {
            item.state: item.annual_absence_days
            for item in value.productivity_loss_states
        }
    return output


def intervention_spec_to_legacy(spec: InterventionSpec) -> dict[str, object]:
    """Return a fresh mutable mapping accepted by the established public API."""
    output: dict[str, object] = {
        "states": list(spec.states),
        "cycles": spec.cycles,
        "initial_population": _vector_list(spec.initial_population),
        "transition_matrices": {
            "standard_care": _matrix_list(spec.transition_matrices.standard_care),
            "new_treatment": _matrix_list(spec.transition_matrices.new_treatment),
        },
        "costs": {
            "health_system": _arm_mapping(spec.costs.health_system),
            "societal": _arm_mapping(spec.costs.societal),
        },
        "qalys": _arm_mapping(spec.qalys),
        "discount_rate": spec.discount_rate,
    }
    if spec.productivity_costs is not None:
        output["productivity_costs"] = {
            "human_capital": _arm_mapping(spec.productivity_costs.human_capital)
        }
    if spec.friction_cost_params is not None:
        output["friction_cost_params"] = spec.friction_cost_params.model_dump()
    if spec.productivity_loss_states:
        output["productivity_loss_states"] = {
            item.state: item.annual_absence_days
            for item in spec.productivity_loss_states
        }
    if spec.subgroups:
        output["subgroups"] = {
            subgroup.name: _subgroup_mapping(subgroup.override)
            for subgroup in spec.subgroups
        }
    return output


def run_typed_cea(
    model_parameters: Mapping[str, Any] | InterventionSpec,
    perspective: str | Perspective = Perspective.HEALTH_SYSTEM,
    wtp_threshold: float = 50_000.0,
    productivity_cost_method: str
    | ProductivityCostMethod = ProductivityCostMethod.HUMAN_CAPITAL,
) -> CEAAnalysisResult:
    """Run CEA through typed contracts without changing the legacy `run_cea`."""
    spec = (
        model_parameters
        if isinstance(model_parameters, InterventionSpec)
        else intervention_spec_from_legacy(model_parameters)
    )
    context = CEACalculationContext(
        perspective=Perspective(perspective),
        wtp_threshold=float(wtp_threshold),
        productivity_cost_method=ProductivityCostMethod(productivity_cost_method),
    )
    return CEACalculationKernel().calculate(spec, context=context)
