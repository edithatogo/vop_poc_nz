"""Generic immutable analysis-result envelopes and metadata."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Protocol, runtime_checkable

from pydantic import Field

from vop_poc_nz.domain.base import FrozenDomainModel
from vop_poc_nz.domain.contracts import ProvenanceSpec


class ResultMaturity(StrEnum):
    """Governed stability level of a result contract."""

    EXPERIMENTAL = "experimental"
    PROVISIONAL = "provisional"
    STABLE = "stable"


class DiagnosticSeverity(StrEnum):
    """Machine-readable severity for calculation diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResultDiagnostic(FrozenDomainModel):
    """Structured calculation diagnostic without free-form mappings."""

    code: str = Field(min_length=1)
    severity: DiagnosticSeverity
    message: str = Field(min_length=1)


class ArrowSchemaIdentity(FrozenDomainModel):
    """Stable identity for the Arrow projection of a result."""

    schema_id: str = Field(min_length=1)
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_logical_fields(
        cls, *, schema_id: str, schema_version: str, logical_fields: tuple[str, ...]
    ) -> ArrowSchemaIdentity:
        canonical = "\n".join(logical_fields).encode("utf-8")
        return cls(
            schema_id=schema_id,
            schema_version=schema_version,
            schema_fingerprint=sha256(canonical).hexdigest(),
        )


class ResultMetadata(FrozenDomainModel):
    """Version, maturity, diagnostics, provenance, and Arrow identity."""

    contract_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    maturity: ResultMaturity
    arrow_schema: ArrowSchemaIdentity
    diagnostics: tuple[ResultDiagnostic, ...] = ()
    provenance: tuple[ProvenanceSpec, ...] = ()


class AnalysisResultEnvelope[PayloadT](FrozenDomainModel):
    """Generic portable result envelope for typed analysis payloads."""

    analysis_type: str = Field(min_length=1)
    metadata: ResultMetadata
    payload: PayloadT


@runtime_checkable
class AnalysisResult(Protocol):
    """Structural contract implemented by concrete analysis results."""

    analysis_type: str
    contract_version: str
    metadata: ResultMetadata

    def to_legacy_dict(self) -> dict[str, object]: ...


__all__ = [
    "AnalysisResult",
    "AnalysisResultEnvelope",
    "ArrowSchemaIdentity",
    "DiagnosticSeverity",
    "ResultDiagnostic",
    "ResultMaturity",
    "ResultMetadata",
]
