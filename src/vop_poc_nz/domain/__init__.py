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

__all__ = [
    "ArmVectors",
    "CostSpec",
    "FrictionCostSpec",
    "FrozenDomainModel",
    "InterventionSpec",
    "NumericVector",
    "Perspective",
    "ProductivityCostMethod",
    "TransitionMatrices",
    "TransitionMatrix",
]
