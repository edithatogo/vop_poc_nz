"""Typed immutable cost-effectiveness results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
from pydantic import Field, model_validator

from vop_poc_nz.domain.base import FrozenDomainModel
from vop_poc_nz.domain.cea import Perspective, ProductivityCostMethod
from vop_poc_nz.domain.contracts import (
    QALY,
    UNKNOWN_CURRENCY,
    ProvenanceSpec,
    UnitSpec,
)
from vop_poc_nz.perspective_io import schema_fingerprint
from vop_poc_nz.results.base import (
    ArrowSchemaIdentity,
    ResultMaturity,
    ResultMetadata,
)

CEA_RESULT_ARROW_SCHEMA = pa.schema(
    (
        pa.field("perspective", pa.string()),
        pa.field("productivity_cost_method", pa.string()),
        pa.field("incremental_cost", pa.float64()),
        pa.field("incremental_qalys", pa.float64()),
        pa.field("incremental_nmb", pa.float64()),
        pa.field("icer_status", pa.string()),
        pa.field("icer_value", pa.float64()),
        pa.field("is_cost_effective", pa.bool_()),
        pa.field("wtp_threshold", pa.float64()),
    )
)


def _cea_metadata(provenance: tuple[ProvenanceSpec, ...] = ()) -> ResultMetadata:
    return ResultMetadata(
        contract_version="1.0.0",
        maturity=ResultMaturity.STABLE,
        arrow_schema=ArrowSchemaIdentity(
            schema_id="cea_analysis_result",
            schema_version="1.0.0",
            schema_fingerprint=schema_fingerprint(CEA_RESULT_ARROW_SCHEMA),
        ),
        provenance=provenance,
    )


class ICERStatus(StrEnum):
    FINITE = "finite"
    DOMINANT = "dominant"
    DOMINATED = "dominated"
    UNDEFINED = "undefined"


class ICERResult(FrozenDomainModel):
    status: ICERStatus
    value: float | None = None

    @model_validator(mode="after")
    def validate_value(self) -> ICERResult:
        if self.status is ICERStatus.FINITE:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("finite ICER status requires a finite value")
        elif self.value is not None:
            raise ValueError("non-finite ICER statuses must not carry a value")
        return self

    @classmethod
    def from_legacy(cls, value: float | str) -> ICERResult:
        if isinstance(value, str):
            return cls(status=ICERStatus.UNDEFINED)
        numeric = float(value)
        if math.isnan(numeric):
            return cls(status=ICERStatus.UNDEFINED)
        if numeric == math.inf:
            return cls(status=ICERStatus.DOMINATED)
        if numeric == -math.inf:
            return cls(status=ICERStatus.DOMINANT)
        return cls(status=ICERStatus.FINITE, value=numeric)

    def to_legacy_value(self) -> float:
        if self.status is ICERStatus.DOMINATED:
            return math.inf
        if self.status is ICERStatus.DOMINANT:
            return -math.inf
        if self.status is ICERStatus.UNDEFINED:
            return math.nan
        if self.value is None:
            raise ValueError("finite ICER results require a numeric value")
        return self.value


Trace = tuple[tuple[float, ...], ...]


class CEAAnalysisResult(FrozenDomainModel):
    analysis_type: ClassVar[str] = "cost_effectiveness"
    contract_version: ClassVar[str] = "1.0.0"

    perspective: Perspective
    productivity_cost_method: ProductivityCostMethod
    cost_standard_care: float
    qalys_standard_care: float
    cost_new_treatment: float
    qalys_new_treatment: float
    incremental_cost: float
    incremental_qalys: float
    icer: ICERResult
    incremental_nmb: float
    is_cost_effective: bool
    wtp_threshold: float
    cost_unit: UnitSpec = UNKNOWN_CURRENCY
    health_outcome_unit: UnitSpec = QALY
    subgroup_results: tuple[NamedSubgroupResult, ...] = ()
    trace_standard_care: Trace | None = None
    trace_new_treatment: Trace | None = None
    metadata: ResultMetadata = Field(default_factory=_cea_metadata)

    @classmethod
    def from_legacy(
        cls,
        result: Mapping[str, Any],
        *,
        cost_unit: UnitSpec = UNKNOWN_CURRENCY,
        health_outcome_unit: UnitSpec = QALY,
        provenance: tuple[ProvenanceSpec, ...] = (),
    ) -> CEAAnalysisResult:
        def trace(value: object) -> Trace | None:
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float64)
            return tuple(tuple(float(item) for item in row) for row in array)

        subgroup_mapping = result.get("subgroup_results") or {}
        subgroups = tuple(
            NamedSubgroupResult(
                name=str(name),
                result=cls.from_legacy(
                    value,
                    cost_unit=cost_unit,
                    health_outcome_unit=health_outcome_unit,
                    provenance=provenance,
                ),
            )
            for name, value in subgroup_mapping.items()
        )
        return cls(
            perspective=Perspective(str(result["perspective"])),
            productivity_cost_method=ProductivityCostMethod(
                str(result["productivity_cost_method"])
            ),
            cost_standard_care=float(result["cost_standard_care"]),
            qalys_standard_care=float(result["qalys_standard_care"]),
            cost_new_treatment=float(result["cost_new_treatment"]),
            qalys_new_treatment=float(result["qalys_new_treatment"]),
            incremental_cost=float(result["incremental_cost"]),
            incremental_qalys=float(result["incremental_qalys"]),
            icer=ICERResult.from_legacy(result["icer"]),
            incremental_nmb=float(result["incremental_nmb"]),
            is_cost_effective=bool(result["is_cost_effective"]),
            wtp_threshold=float(result["wtp_threshold"]),
            cost_unit=cost_unit,
            health_outcome_unit=health_outcome_unit,
            subgroup_results=subgroups,
            trace_standard_care=trace(result.get("trace_standard_care")),
            trace_new_treatment=trace(result.get("trace_new_treatment")),
            metadata=_cea_metadata(provenance),
        )

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "perspective": self.perspective.value,
            "cost_standard_care": self.cost_standard_care,
            "qalys_standard_care": self.qalys_standard_care,
            "cost_new_treatment": self.cost_new_treatment,
            "qalys_new_treatment": self.qalys_new_treatment,
            "incremental_cost": self.incremental_cost,
            "incremental_qalys": self.incremental_qalys,
            "icer": self.icer.to_legacy_value(),
            "incremental_nmb": self.incremental_nmb,
            "is_cost_effective": self.is_cost_effective,
            "wtp_threshold": self.wtp_threshold,
            "productivity_cost_method": self.productivity_cost_method.value,
            "subgroup_results": {
                subgroup.name: subgroup.result.to_legacy_dict()
                for subgroup in self.subgroup_results
            },
            "trace_standard_care": None
            if self.trace_standard_care is None
            else np.asarray(self.trace_standard_care, dtype=np.float64),
            "trace_new_treatment": None
            if self.trace_new_treatment is None
            else np.asarray(self.trace_new_treatment, dtype=np.float64),
        }


class NamedSubgroupResult(FrozenDomainModel):
    name: str
    result: CEAAnalysisResult


CEAAnalysisResult.model_rebuild()
