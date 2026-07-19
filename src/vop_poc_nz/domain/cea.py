"""Immutable domain specifications for cost-effectiveness analysis."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .base import FrozenDomainModel
from .contracts import CYCLE, NZD_UNSPECIFIED, PERSON, QALY, ProvenanceSpec, UnitSpec


class Perspective(StrEnum):
    HEALTH_SYSTEM = "health_system"
    SOCIETAL = "societal"


class ProductivityCostMethod(StrEnum):
    HUMAN_CAPITAL = "human_capital"
    FRICTION_COST = "friction_cost"


class NumericVector(FrozenDomainModel):
    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def values_are_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite")
        return values


class TransitionMatrix(FrozenDomainModel):
    rows: tuple[tuple[float, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> TransitionMatrix:
        size = len(self.rows)
        if any(len(row) != size for row in self.rows):
            raise ValueError("transition matrix must be square")
        for row in self.rows:
            if not all(math.isfinite(value) and value >= 0.0 for value in row):
                raise ValueError(
                    "transition probabilities must be finite and non-negative"
                )
            if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("transition matrix rows must sum to 1")
        return self


class TransitionMatrices(FrozenDomainModel):
    standard_care: TransitionMatrix
    new_treatment: TransitionMatrix


class PartialTransitionMatrices(FrozenDomainModel):
    standard_care: TransitionMatrix | None = None
    new_treatment: TransitionMatrix | None = None


class ArmVectors(FrozenDomainModel):
    standard_care: NumericVector
    new_treatment: NumericVector


class PartialArmVectors(FrozenDomainModel):
    standard_care: NumericVector | None = None
    new_treatment: NumericVector | None = None


class CostSpec(FrozenDomainModel):
    health_system: ArmVectors
    societal: ArmVectors


class PartialCostSpec(FrozenDomainModel):
    health_system: PartialArmVectors | None = None
    societal: PartialArmVectors | None = None


class ProductivityCostSpec(FrozenDomainModel):
    human_capital: ArmVectors


class FrictionCostSpec(FrozenDomainModel):
    friction_period_days: float = Field(ge=0.0)
    replacement_cost_per_day: float = Field(ge=0.0)
    absenteeism_rate: float = Field(ge=0.0, le=1.0)


class ProductivityLossState(FrozenDomainModel):
    state: str = Field(min_length=1)
    annual_absence_days: float = Field(ge=0.0)


class SubgroupOverrideSpec(FrozenDomainModel):
    initial_population: NumericVector | None = None
    transition_matrices: PartialTransitionMatrices | None = None
    costs: PartialCostSpec | None = None
    qalys: PartialArmVectors | None = None
    discount_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    productivity_costs: ProductivityCostSpec | None = None
    friction_cost_params: FrictionCostSpec | None = None
    productivity_loss_states: tuple[ProductivityLossState, ...] | None = None


class NamedSubgroupSpec(FrozenDomainModel):
    name: str = Field(min_length=1)
    override: SubgroupOverrideSpec


def _present_partial_vectors(
    value: PartialArmVectors | None,
) -> tuple[NumericVector, ...]:
    if value is None:
        return ()
    return tuple(
        vector
        for vector in (value.standard_care, value.new_treatment)
        if vector is not None
    )


def _subgroup_vectors(value: SubgroupOverrideSpec) -> tuple[NumericVector, ...]:
    vectors = list(_present_partial_vectors(value.qalys))
    if value.costs is not None:
        vectors.extend(_present_partial_vectors(value.costs.health_system))
        vectors.extend(_present_partial_vectors(value.costs.societal))
    if value.productivity_costs is not None:
        vectors.extend(
            (
                value.productivity_costs.human_capital.standard_care,
                value.productivity_costs.human_capital.new_treatment,
            )
        )
    return tuple(vectors)


class InterventionSpec(FrozenDomainModel):
    """Validated inputs consumed by the typed CEA calculation kernel."""

    states: tuple[str, ...] = Field(min_length=1)
    cycles: int = Field(gt=0)
    cycle_unit: UnitSpec = CYCLE
    initial_population: NumericVector
    population_unit: UnitSpec = PERSON
    transition_matrices: TransitionMatrices
    costs: CostSpec
    cost_unit: UnitSpec = NZD_UNSPECIFIED
    qalys: ArmVectors
    health_outcome_unit: UnitSpec = QALY
    discount_rate: float = Field(default=0.03, ge=0.0, le=1.0)
    productivity_costs: ProductivityCostSpec | None = None
    friction_cost_params: FrictionCostSpec | None = None
    productivity_loss_states: tuple[ProductivityLossState, ...] = ()
    subgroups: tuple[NamedSubgroupSpec, ...] = ()
    provenance: tuple[ProvenanceSpec, ...] = ()

    @field_validator("states")
    @classmethod
    def states_are_unique(cls, states: tuple[str, ...]) -> tuple[str, ...]:
        if any(not state for state in states):
            raise ValueError("state names must not be empty")
        if len(set(states)) != len(states):
            raise ValueError("state names must be unique")
        return states

    @model_validator(mode="after")
    def validate_dimensions(self) -> InterventionSpec:
        size = len(self.states)
        vectors = [
            self.initial_population,
            self.costs.health_system.standard_care,
            self.costs.health_system.new_treatment,
            self.costs.societal.standard_care,
            self.costs.societal.new_treatment,
            self.qalys.standard_care,
            self.qalys.new_treatment,
        ]
        if self.productivity_costs is not None:
            vectors.extend(
                (
                    self.productivity_costs.human_capital.standard_care,
                    self.productivity_costs.human_capital.new_treatment,
                )
            )
        if any(len(vector.values) != size for vector in vectors):
            raise ValueError("all vectors must have one value per state")
        for matrix in (
            self.transition_matrices.standard_care,
            self.transition_matrices.new_treatment,
        ):
            if len(matrix.rows) != size:
                raise ValueError("transition matrices must match the state count")
        if len({subgroup.name for subgroup in self.subgroups}) != len(self.subgroups):
            raise ValueError("subgroup names must be unique")
        self._validate_subgroups(size)
        return self

    def _validate_subgroups(self, size: int) -> None:
        for named in self.subgroups:
            override = named.override
            if (
                override.initial_population is not None
                and len(override.initial_population.values) != size
            ):
                raise ValueError(
                    f"subgroup {named.name!r} population must match the state count"
                )
            matrices = override.transition_matrices
            if matrices is not None:
                for matrix in (matrices.standard_care, matrices.new_treatment):
                    if matrix is not None and len(matrix.rows) != size:
                        raise ValueError(
                            f"subgroup {named.name!r} matrix must match the state count"
                        )
            if any(
                len(vector.values) != size for vector in _subgroup_vectors(override)
            ):
                raise ValueError(
                    f"subgroup {named.name!r} vectors must match the state count"
                )
