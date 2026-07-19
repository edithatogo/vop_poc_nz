"""Opt-in typed, calculation-only analysis pipeline.

The established ``run_analysis_pipeline`` remains the legacy orchestration
entrypoint. This module deliberately performs no reporting or artifact I/O.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import Field, field_validator

from vop_poc_nz.compat.legacy import (
    intervention_spec_from_legacy,
    run_typed_cea,
)
from vop_poc_nz.domain.base import FrozenDomainModel
from vop_poc_nz.domain.cea import (
    InterventionSpec,
    Perspective,
    ProductivityCostMethod,
)
from vop_poc_nz.logging_config import log_context
from vop_poc_nz.results.cea import CEAAnalysisResult
from vop_poc_nz.results.pipeline import (
    InterventionPipelineResult,
    SocietalCEAResult,
    TypedPipelineResult,
)

logger = logging.getLogger(__name__)


class NamedInterventionSpec(FrozenDomainModel):
    name: str = Field(min_length=1)
    spec: InterventionSpec


class TypedPipelineSpec(FrozenDomainModel):
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    random_seed: int | None = None
    software_version: str | None = None
    spec_fingerprint: str = Field(min_length=64, max_length=64)
    interventions: tuple[NamedInterventionSpec, ...] = Field(min_length=1)

    @field_validator("created_at_utc")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)


def _fingerprint(interventions: tuple[NamedInterventionSpec, ...]) -> str:
    payload = [
        {"name": intervention.name, "spec": intervention.spec.model_dump(mode="json")}
        for intervention in interventions
    ]
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def typed_pipeline_spec_from_legacy(
    interventions: Mapping[str, Mapping[str, Any]],
    *,
    run_id: str,
    created_at_utc: datetime | None = None,
    random_seed: int | None = None,
    software_version: str | None = None,
) -> TypedPipelineSpec:
    """Validate and deep-freeze legacy intervention mappings without mutation."""
    typed = tuple(
        NamedInterventionSpec(
            name=str(name), spec=intervention_spec_from_legacy(parameters)
        )
        for name, parameters in interventions.items()
    )
    return TypedPipelineSpec(
        run_id=run_id,
        created_at_utc=created_at_utc or datetime.now(UTC),
        random_seed=random_seed,
        software_version=software_version,
        spec_fingerprint=_fingerprint(typed),
        interventions=typed,
    )


def _calculate_intervention(
    item: NamedInterventionSpec,
    *,
    pipeline: TypedPipelineSpec,
) -> InterventionPipelineResult:
    with log_context(
        pipeline_run_id=pipeline.run_id,
        intervention=item.name,
        spec_fingerprint=pipeline.spec_fingerprint,
        stage="calculation",
    ):
        logger.info("typed_cea_calculation_started")
        health_system = run_typed_cea(
            item.spec,
            perspective=Perspective.HEALTH_SYSTEM,
        )
        societal = tuple(
            SocietalCEAResult(
                method=method,
                result=run_typed_cea(
                    item.spec,
                    perspective=Perspective.SOCIETAL,
                    productivity_cost_method=method,
                ),
            )
            for method in (
                ProductivityCostMethod.HUMAN_CAPITAL,
                ProductivityCostMethod.FRICTION_COST,
            )
        )
        logger.info("typed_cea_calculation_completed")
    return InterventionPipelineResult(
        name=item.name,
        health_system=health_system,
        societal=societal,
    )


def run_typed_analysis_pipeline(spec: TypedPipelineSpec) -> TypedPipelineResult:
    """Calculate immutable CEA bundles without reporting or artifact writes."""
    return TypedPipelineResult(
        run_id=spec.run_id,
        created_at_utc=spec.created_at_utc,
        random_seed=spec.random_seed,
        software_version=spec.software_version,
        spec_fingerprint=spec.spec_fingerprint,
        interventions=tuple(
            _calculate_intervention(intervention, pipeline=spec)
            for intervention in spec.interventions
        ),
    )


def _result_record(
    *,
    pipeline: TypedPipelineResult,
    intervention: str,
    result: CEAAnalysisResult,
) -> dict[str, object]:
    return {
        "run_id": pipeline.run_id,
        "created_at_utc": pipeline.created_at_utc.isoformat(),
        "software_version": pipeline.software_version,
        "random_seed": pipeline.random_seed,
        "spec_fingerprint": pipeline.spec_fingerprint,
        "contract_version": pipeline.contract_version,
        "intervention": intervention,
        "perspective": result.perspective.value,
        "productivity_cost_method": result.productivity_cost_method.value,
        "incremental_cost": result.incremental_cost,
        "incremental_qalys": result.incremental_qalys,
        "incremental_nmb": result.incremental_nmb,
        "icer_status": result.icer.status.value,
        "icer_value": result.icer.value,
        "is_cost_effective": result.is_cost_effective,
        "wtp_threshold": result.wtp_threshold,
    }


def pipeline_result_records(result: TypedPipelineResult) -> list[dict[str, object]]:
    """Adapt a typed bundle to Arrow-ready scalar records without performing I/O."""
    records: list[dict[str, object]] = []
    for intervention in result.interventions:
        records.append(
            _result_record(
                pipeline=result,
                intervention=intervention.name,
                result=intervention.health_system,
            )
        )
        records.extend(
            _result_record(
                pipeline=result,
                intervention=intervention.name,
                result=societal.result,
            )
            for societal in intervention.societal
        )
    return records
