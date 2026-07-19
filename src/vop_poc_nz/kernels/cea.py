"""Typed CEA kernel over the established, characterized calculation path."""

from __future__ import annotations

from typing import ClassVar

from vop_poc_nz.domain.cea import (
    InterventionSpec,
    Perspective,
    ProductivityCostMethod,
)
from vop_poc_nz.results.cea import CEAAnalysisResult

from .base import CalculationContext


class CEACalculationContext(CalculationContext):
    perspective: Perspective = Perspective.HEALTH_SYSTEM
    wtp_threshold: float = 50_000.0
    productivity_cost_method: ProductivityCostMethod = (
        ProductivityCostMethod.HUMAN_CAPITAL
    )


class CEACalculationKernel:
    """Produce an immutable result while preserving legacy numerical semantics."""

    name: ClassVar[str] = "cea"
    contract_version: ClassVar[str] = "1.0.0"

    def calculate(
        self, spec: InterventionSpec, *, context: CEACalculationContext
    ) -> CEAAnalysisResult:
        # Import locally to keep the domain/result layers independent of the
        # compatibility implementation and prevent an import cycle.
        from vop_poc_nz.cea_model_core import run_cea
        from vop_poc_nz.compat.legacy import intervention_spec_to_legacy

        legacy = run_cea(
            intervention_spec_to_legacy(spec),
            perspective=context.perspective.value,
            wtp_threshold=context.wtp_threshold,
            productivity_cost_method=context.productivity_cost_method.value,
        )
        return CEAAnalysisResult.from_legacy(legacy)
