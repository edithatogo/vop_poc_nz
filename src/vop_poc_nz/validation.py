"""
Validation utilities using pandera and pydantic for health economic analyses.
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from pandera import Column, DataFrameSchema
from pydantic import BaseModel, Field

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "true")

PSAResultsSchema = DataFrameSchema(
    {
        "qaly_sc": Column(float),
        "qaly_nt": Column(float),
        "cost_sc": Column(float),
        "cost_nt": Column(float),
    },
    strict=False,
    coerce=True,
)


def validate_psa_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate PSA results DataFrame to ensure required columns and numeric types.
    """
    return PSAResultsSchema.validate(df)


class ParametersModel(BaseModel):
    states: list[str] = Field(min_length=1)
    cycles: int = Field(gt=0)
    initial_population: list[int] = Field(min_length=1)
    discount_rate: float = Field(0.03, ge=0, le=1)
    costs: dict
    qalys: dict
    transition_matrices: dict
    productivity_costs: Optional[dict] = None
    friction_cost_params: Optional[dict] = None
    productivity_loss_states: Optional[dict] = None


def validate_transition_matrices(params: dict) -> None:
    """Validate transition matrices for shape, non-negativity, and row sums."""
    states = params.get("states", [])
    tm = params.get("transition_matrices", {})
    for name in ("standard_care", "new_treatment"):
        matrix = np.array(tm.get(name, []), dtype=float)
        if matrix.shape != (len(states), len(states)):
            raise ValueError(
                f"Transition matrix '{name}' must be square with size len(states)={len(states)}"
            )
        if np.any(matrix < 0):
            raise ValueError(f"Transition matrix '{name}' contains negative entries")
        row_sums = np.sum(matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Transition matrix '{name}' rows must sum to 1.0; got {row_sums}"
            )


def validate_costs_and_qalys(params: dict) -> None:
    """Ensure cost and QALY arrays are present and non-negative."""
    for perspective in ("health_system", "societal"):
        costs = params.get("costs", {}).get(perspective, {})
        for arm in ("standard_care", "new_treatment"):
            arr = np.array(costs.get(arm, []), dtype=float)
            if np.any(arr < 0):
                warnings.warn(
                    f"Costs for {perspective}/{arm} contain negative values (interpreted as savings).",
                    UserWarning,
                    stacklevel=2,
                )
    for arm in ("standard_care", "new_treatment"):
        qalys = np.array(params.get("qalys", {}).get(arm, []), dtype=float)
        if np.any(qalys < 0):
            raise ValueError(f"QALYs for {arm} contain negative values")


__all__ = [
    "PSAResultsSchema",
    "ParametersModel",
    "validate_costs_and_qalys",
    "validate_psa_results",
    "validate_transition_matrices",
]
