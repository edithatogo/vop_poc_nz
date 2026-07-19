"""Typed CEA kernel over the established, characterized calculation path."""

from __future__ import annotations

import warnings
from typing import ClassVar, cast

from vop_poc_nz.domain.cea import (
    InterventionSpec,
    Perspective,
    ProductivityCostMethod,
)
from vop_poc_nz.domain.contracts import QALY, UNKNOWN_CURRENCY, UnitSpec
from vop_poc_nz.results.base import DiagnosticSeverity, ResultDiagnostic
from vop_poc_nz.results.cea import CEAAnalysisResult

from .base import CalculationContext


def _core_value(value: object) -> object:
    """Project typed models to the isolated numerical core's input shape."""
    if isinstance(value, dict):
        vector = value.get("values")
        if set(value) == {"values"} and isinstance(vector, (list, tuple)):
            return list(vector)
        rows = value.get("rows")
        if set(value) == {"rows"} and isinstance(rows, (list, tuple)):
            return [list(row) for row in rows if isinstance(row, (list, tuple))]
        return {
            str(key): _core_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, tuple):
        return [_core_value(item) for item in value]
    return value


def _core_mapping(spec: InterventionSpec) -> dict[str, object]:
    raw = spec.model_dump(
        mode="python",
        exclude={
            "cycle_unit",
            "population_unit",
            "cost_unit",
            "health_outcome_unit",
            "provenance",
        },
    )
    projected = cast(dict[str, object], _core_value(raw))
    projected["productivity_loss_states"] = {
        item.state: item.annual_absence_days for item in spec.productivity_loss_states
    }
    projected["subgroups"] = {
        item.name: _core_value(item.override.model_dump(mode="python"))
        for item in spec.subgroups
    }
    if not projected["productivity_loss_states"]:
        projected.pop("productivity_loss_states")
    if not projected["subgroups"]:
        projected.pop("subgroups")
    return projected


class CEACalculationContext(CalculationContext):
    perspective: Perspective = Perspective.HEALTH_SYSTEM
    wtp_threshold: float = 50_000.0
    productivity_cost_method: ProductivityCostMethod = (
        ProductivityCostMethod.HUMAN_CAPITAL
    )
    cost_unit: UnitSpec = UNKNOWN_CURRENCY
    health_outcome_unit: UnitSpec = QALY


class CEACalculationKernel:
    """Produce an immutable result while preserving legacy numerical semantics."""

    name: ClassVar[str] = "cea"
    contract_version: ClassVar[str] = "1.0.0"

    def calculate(
        self, spec: InterventionSpec, *, context: CEACalculationContext
    ) -> CEAAnalysisResult:
        from vop_poc_nz.cea_model_core import calculate_cea

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = calculate_cea(
                _core_mapping(spec),
                perspective=context.perspective.value,
                wtp_threshold=context.wtp_threshold,
                productivity_cost_method=context.productivity_cost_method.value,
            )
        if context.cost_unit != spec.cost_unit:
            raise ValueError("context cost_unit must match intervention cost_unit")
        if context.health_outcome_unit != spec.health_outcome_unit:
            raise ValueError(
                "context health_outcome_unit must match intervention health_outcome_unit"
            )
        result = CEAAnalysisResult.from_legacy(
            legacy,
            cost_unit=spec.cost_unit,
            health_outcome_unit=spec.health_outcome_unit,
            provenance=spec.provenance,
        )
        diagnostics = tuple(
            ResultDiagnostic(
                code=f"PYTHON_WARNING_{warning.category.__name__.upper()}",
                severity=DiagnosticSeverity.WARNING,
                message=str(warning.message),
            )
            for warning in caught
        )
        metadata = result.metadata.model_copy(
            update={
                "diagnostics": (*result.metadata.diagnostics, *diagnostics),
                "provenance": spec.provenance,
            }
        )
        return result.model_copy(update={"metadata": metadata})
