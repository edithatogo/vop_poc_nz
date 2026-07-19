"""Portable parameter, uncertainty, numerical-policy, and run contracts."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .base import FrozenDomainModel


class UnitDimension(StrEnum):
    """Physical or decision-model dimension represented by a unit."""

    CURRENCY = "currency"
    TIME = "time"
    HEALTH = "health"
    PROBABILITY = "probability"
    COUNT = "count"
    DIMENSIONLESS = "dimensionless"


class UnitSpec(FrozenDomainModel):
    """Machine-readable unit identity, including currency price year."""

    symbol: str = Field(min_length=1)
    dimension: UnitDimension
    currency_code: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    currency_year: int | None = Field(default=None, ge=1900, le=2200)

    @model_validator(mode="after")
    def currency_metadata_is_consistent(self) -> UnitSpec:
        is_currency = self.dimension is UnitDimension.CURRENCY
        if is_currency != (self.currency_code is not None):
            raise ValueError("currency units require an ISO currency_code")
        if not is_currency and self.currency_year is not None:
            raise ValueError("currency_year is only valid for currency units")
        return self


class ProvenanceSpec(FrozenDomainModel):
    """Traceable identity for an input, source, or generated contract."""

    source_id: str = Field(min_length=1)
    observed_at_utc: datetime | None = None
    source_version: str | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("observed_at_utc")
    @classmethod
    def timestamp_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at_utc must be timezone-aware")
        return value.astimezone(UTC)


class ParameterSpec(FrozenDomainModel):
    """Named scalar parameter with unit, dimensions, and provenance."""

    name: str = Field(min_length=1)
    value: float
    unit: UnitSpec
    dimensions: tuple[str, ...] = ()
    provenance: ProvenanceSpec

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("parameter value must be finite")
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError("parameter dimensions must be non-empty and unique")
        return values


class DistributionFamily(StrEnum):
    """Supported uncertainty distribution families."""

    FIXED = "fixed"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    UNIFORM = "uniform"


class DistributionParameter(FrozenDomainModel):
    """Named finite parameter of an uncertainty distribution."""

    name: str = Field(min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("distribution parameters must be finite")
        return value


class DistributionSpec(FrozenDomainModel):
    """Validated uncertainty distribution with explicit parameter names."""

    family: DistributionFamily
    parameters: tuple[DistributionParameter, ...] = Field(min_length=1)
    unit: UnitSpec
    provenance: ProvenanceSpec

    @model_validator(mode="after")
    def parameters_match_family(self) -> DistributionSpec:
        values = {item.name: item.value for item in self.parameters}
        if len(values) != len(self.parameters):
            raise ValueError("distribution parameter names must be unique")
        required = {
            DistributionFamily.FIXED: {"value"},
            DistributionFamily.NORMAL: {"mean", "standard_deviation"},
            DistributionFamily.LOGNORMAL: {"meanlog", "sdlog"},
            DistributionFamily.BETA: {"alpha", "beta"},
            DistributionFamily.GAMMA: {"shape", "scale"},
            DistributionFamily.UNIFORM: {"low", "high"},
        }[self.family]
        if set(values) != required:
            raise ValueError(
                f"{self.family.value} requires parameters {sorted(required)}"
            )
        if self.family in {DistributionFamily.BETA, DistributionFamily.GAMMA} and any(
            value <= 0.0 for value in values.values()
        ):
            raise ValueError("beta and gamma parameters must be positive")
        if (
            self.family is DistributionFamily.NORMAL
            and values["standard_deviation"] <= 0
        ):
            raise ValueError("standard_deviation must be positive")
        if self.family is DistributionFamily.LOGNORMAL and values["sdlog"] <= 0:
            raise ValueError("sdlog must be positive")
        if (
            self.family is DistributionFamily.UNIFORM
            and values["low"] >= values["high"]
        ):
            raise ValueError("uniform low must be less than high")
        return self


class NonFinitePolicy(StrEnum):
    """Permitted response to non-finite numerical output."""

    RAISE = "raise"
    REPORT = "report"


class NumericalPolicySpec(FrozenDomainModel):
    """Explicit reproducibility and convergence policy for a calculation."""

    absolute_tolerance: float = Field(default=1e-9, gt=0.0)
    relative_tolerance: float = Field(default=1e-9, gt=0.0)
    max_iterations: int = Field(default=10_000, gt=0)
    non_finite_policy: NonFinitePolicy = NonFinitePolicy.RAISE
    deterministic: bool = True


class AnalysisSpec(FrozenDomainModel):
    """Generic analysis identity and its typed scalar inputs."""

    analysis_type: str = Field(min_length=1)
    contract_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    parameters: tuple[ParameterSpec, ...] = ()
    numerical_policy: NumericalPolicySpec = NumericalPolicySpec()

    @model_validator(mode="after")
    def parameter_names_are_unique(self) -> AnalysisSpec:
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("analysis parameter names must be unique")
        return self


class RunContextSpec(FrozenDomainModel):
    """Portable execution context shared by deterministic analysis kernels."""

    run_id: str | None = None
    case_id: str | None = None
    created_at_utc: datetime | None = None
    seed: int | None = None
    software_version: str | None = None
    numerical_policy: NumericalPolicySpec = NumericalPolicySpec()
    provenance: tuple[ProvenanceSpec, ...] = ()

    @field_validator("created_at_utc")
    @classmethod
    def timestamp_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)


NZD_UNSPECIFIED = UnitSpec(
    symbol="NZD",
    dimension=UnitDimension.CURRENCY,
    currency_code="NZD",
)
QALY = UnitSpec(symbol="QALY", dimension=UnitDimension.HEALTH)
PERSON = UnitSpec(symbol="person", dimension=UnitDimension.COUNT)
CYCLE = UnitSpec(symbol="cycle", dimension=UnitDimension.TIME)


__all__ = [
    "CYCLE",
    "NZD_UNSPECIFIED",
    "PERSON",
    "QALY",
    "AnalysisSpec",
    "DistributionFamily",
    "DistributionParameter",
    "DistributionSpec",
    "NonFinitePolicy",
    "NumericalPolicySpec",
    "ParameterSpec",
    "ProvenanceSpec",
    "RunContextSpec",
    "UnitDimension",
    "UnitSpec",
]
