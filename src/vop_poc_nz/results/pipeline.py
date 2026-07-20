"""Immutable result bundles for the opt-in typed analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from vop_poc_nz.domain.base import FrozenDomainModel
from vop_poc_nz.domain.cea import ProductivityCostMethod
from vop_poc_nz.results.base import ResultMetadata
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

    contract_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    random_seed: int | None = None
    software_version: str | None = None
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    interventions: tuple[InterventionPipelineResult, ...]
    metadata: ResultMetadata

    @model_validator(mode="after")
    def intervention_names_are_unique(self) -> TypedPipelineResult:
        names = tuple(item.name for item in self.interventions)
        if len(set(names)) != len(names):
            raise ValueError("intervention result names must be unique")
        return self

    def to_legacy_intervention_results(self) -> dict[str, dict[str, object]]:
        """Reconstruct the established CEA portion of `intervention_results`."""
        return {
            intervention.name: intervention.to_legacy_dict()
            for intervention in self.interventions
        }
