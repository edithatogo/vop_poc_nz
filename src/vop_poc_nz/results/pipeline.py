"""Immutable result bundles for the opt-in typed analysis pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from vop_poc_nz.domain.base import FrozenDomainModel
from vop_poc_nz.domain.cea import ProductivityCostMethod
from vop_poc_nz.results.cea import CEAAnalysisResult


class SocietalCEAResult(FrozenDomainModel):
    method: ProductivityCostMethod
    result: CEAAnalysisResult


class InterventionPipelineResult(FrozenDomainModel):
    name: str = Field(min_length=1)
    health_system: CEAAnalysisResult
    societal: tuple[SocietalCEAResult, ...]

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "health_system": self.health_system.to_legacy_dict(),
            "societal": {
                item.method.value: item.result.to_legacy_dict()
                for item in self.societal
            },
        }


class TypedPipelineResult(FrozenDomainModel):
    """Calculation-only output; reporting and artifact I/O are external adapters."""

    contract_version: str = "1.0.0"
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    random_seed: int | None = None
    software_version: str | None = None
    spec_fingerprint: str = Field(min_length=64, max_length=64)
    interventions: tuple[InterventionPipelineResult, ...]

    def to_legacy_intervention_results(self) -> dict[str, dict[str, object]]:
        """Reconstruct the established CEA portion of `intervention_results`."""
        return {
            intervention.name: intervention.to_legacy_dict()
            for intervention in self.interventions
        }
