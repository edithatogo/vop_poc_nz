"""Immutable domain specifications for health-economic calculations."""

from .base import FrozenDomainModel
from .cea import (
    ArmVectors,
    CostSpec,
    FrictionCostSpec,
    InterventionSpec,
    NumericVector,
    Perspective,
    ProductivityCostMethod,
    TransitionMatrices,
    TransitionMatrix,
)
from .contracts import (
    AnalysisSpec,
    DistributionFamily,
    DistributionParameter,
    DistributionSpec,
    NonFinitePolicy,
    NumericalPolicySpec,
    ParameterSpec,
    ProvenanceSpec,
    RunContextSpec,
    UnitDimension,
    UnitSpec,
)

__all__ = [
    "AnalysisSpec",
    "ArmVectors",
    "CostSpec",
    "DistributionFamily",
    "DistributionParameter",
    "DistributionSpec",
    "FrictionCostSpec",
    "FrozenDomainModel",
    "InterventionSpec",
    "NonFinitePolicy",
    "NumericVector",
    "NumericalPolicySpec",
    "ParameterSpec",
    "Perspective",
    "ProductivityCostMethod",
    "ProvenanceSpec",
    "RunContextSpec",
    "TransitionMatrices",
    "TransitionMatrix",
    "UnitDimension",
    "UnitSpec",
]
