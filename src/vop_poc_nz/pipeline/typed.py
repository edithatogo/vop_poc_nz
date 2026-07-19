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

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from vop_poc_nz.compat.legacy import (
    intervention_spec_from_legacy,
    run_typed_cea,
)
from vop_poc_nz.critical_invariants import (
    require_matching_sha256,
    supported_societal_methods,
)
from vop_poc_nz.domain.base import FrozenDomainModel
from vop_poc_nz.domain.cea import (
    InterventionSpec,
    Perspective,
    ProductivityCostMethod,
)
from vop_poc_nz.domain.contracts import NumericalPolicySpec, ProvenanceSpec
from vop_poc_nz.logging_config import (
    AnalysisLogContext,
    analysis_log_context,
    log_context,
    numerical_policy_digest,
)
from vop_poc_nz.perspective_io import attach_contract_metadata, schema_fingerprint
from vop_poc_nz.results.base import ArrowSchemaIdentity, ResultMaturity, ResultMetadata
from vop_poc_nz.results.cea import CEAAnalysisResult
from vop_poc_nz.results.pipeline import (
    InterventionPipelineResult,
    SocietalCEAResult,
    TypedPipelineResult,
)

logger = logging.getLogger(__name__)

TYPED_PIPELINE_ARROW_SCHEMA = pa.schema(
    (
        pa.field("run_id", pa.string()),
        pa.field("created_at_utc", pa.string()),
        pa.field("software_version", pa.string()),
        pa.field("random_seed", pa.int64()),
        pa.field("spec_fingerprint", pa.string()),
        pa.field("contract_version", pa.string()),
        pa.field("intervention", pa.string()),
        pa.field("perspective", pa.string()),
        pa.field("productivity_cost_method", pa.string()),
        pa.field("incremental_cost", pa.float64()),
        pa.field("incremental_qalys", pa.float64()),
        pa.field("incremental_nmb", pa.float64()),
        pa.field("icer_status", pa.string()),
        pa.field("icer_value", pa.float64()),
        pa.field("is_cost_effective", pa.bool_()),
        pa.field("wtp_threshold", pa.float64()),
        pa.field("cost_unit", pa.string()),
        pa.field("cost_currency_code", pa.string()),
        pa.field("cost_currency_year", pa.int64()),
        pa.field("health_outcome_unit", pa.string()),
    )
)


class NamedInterventionSpec(FrozenDomainModel):
    name: str = Field(min_length=1)
    spec: InterventionSpec


class TypedPipelineSpec(FrozenDomainModel):
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    random_seed: int | None = None
    software_version: str | None = None
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    interventions: tuple[NamedInterventionSpec, ...] = Field(min_length=1)

    @field_validator("created_at_utc")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_identity(self) -> TypedPipelineSpec:
        """Reject duplicate names and caller-forged provenance fingerprints."""
        names = tuple(item.name for item in self.interventions)
        if len(set(names)) != len(names):
            raise ValueError("intervention names must be unique")
        require_matching_sha256(
            declared=self.spec_fingerprint,
            actual=_fingerprint(self.interventions),
            field="spec_fingerprint",
        )
        return self


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
        supported_methods = tuple(
            ProductivityCostMethod(method)
            for method in supported_societal_methods(
                has_human_capital=item.spec.productivity_costs is not None,
                has_friction_cost=item.spec.friction_cost_params is not None,
            )
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
            for method in supported_methods
        )
        logger.info("typed_cea_calculation_completed")
    return InterventionPipelineResult(
        name=item.name,
        health_system=health_system,
        societal=societal,
    )


def run_typed_analysis_pipeline(spec: TypedPipelineSpec) -> TypedPipelineResult:
    """Calculate immutable CEA bundles without reporting or artifact writes."""
    numerical_policy = NumericalPolicySpec()
    correlation = AnalysisLogContext(
        run_id=spec.run_id,
        analysis_id="typed-cea",
        backend_requested="numpy",
        backend_selected="numpy",
        fallback_code="none",
        numerical_policy_id=numerical_policy_digest(numerical_policy),
    )
    with analysis_log_context(correlation):
        return TypedPipelineResult(
            run_id=spec.run_id,
            created_at_utc=spec.created_at_utc,
            random_seed=spec.random_seed,
            software_version=spec.software_version,
            spec_fingerprint=spec.spec_fingerprint,
            metadata=ResultMetadata(
                contract_version="1.0.0",
                maturity=ResultMaturity.STABLE,
                arrow_schema=ArrowSchemaIdentity(
                    schema_id="typed_pipeline_records",
                    schema_version="1.0.0",
                    schema_fingerprint=schema_fingerprint(TYPED_PIPELINE_ARROW_SCHEMA),
                ),
                provenance=(
                    ProvenanceSpec(
                        source_id=f"typed-pipeline:{spec.run_id}",
                        observed_at_utc=spec.created_at_utc,
                        source_version=spec.software_version,
                        content_sha256=spec.spec_fingerprint,
                    ),
                ),
            ),
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
        "cost_unit": result.cost_unit.symbol,
        "cost_currency_code": result.cost_unit.currency_code,
        "cost_currency_year": result.cost_unit.currency_year,
        "health_outcome_unit": result.health_outcome_unit.symbol,
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


def pipeline_result_arrow_table(result: TypedPipelineResult) -> pa.Table:
    """Project a result through the canonical schema named by its metadata."""
    table = pa.Table.from_pylist(
        pipeline_result_records(result), schema=TYPED_PIPELINE_ARROW_SCHEMA
    )
    provenance_json = json.dumps(
        [item.model_dump(mode="json") for item in result.metadata.provenance],
        sort_keys=True,
        separators=(",", ":"),
    )
    return attach_contract_metadata(
        table,
        schema_id=result.metadata.arrow_schema.schema_id,
        schema_version=result.metadata.arrow_schema.schema_version,
        contract_version=result.metadata.contract_version,
        expected_fingerprint=result.metadata.arrow_schema.schema_fingerprint,
        provenance_json=provenance_json,
    )
