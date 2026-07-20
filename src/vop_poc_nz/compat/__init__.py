"""Explicit compatibility adapters for legacy public shapes."""

from .legacy import (
    intervention_spec_from_legacy,
    intervention_spec_to_legacy,
    run_typed_cea,
)

__all__ = [
    "intervention_spec_from_legacy",
    "intervention_spec_to_legacy",
    "run_typed_cea",
]
