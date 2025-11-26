"""
Validation utilities using pandera and pydantic for health economic analyses.
"""

from typing import Any, List, Optional

import os

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "true")

from pandera import Column, DataFrameSchema
import pandas as pd
from pydantic import BaseModel, Field

PSAResultsSchema = DataFrameSchema(
    {
        "qaly_sc": Column(float),
        "qaly_nt": Column(float),
        "cost_sc": Column(float),
        "cost_nt": Column(float),
    },
    strict=True,
    coerce=True,
)


def validate_psa_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate PSA results DataFrame to ensure required columns and numeric types.
    """
    return PSAResultsSchema.validate(df)


class ParametersModel(BaseModel):
    states: List[str] = Field(min_length=1)
    cycles: int = Field(gt=0)
    initial_population: List[int] = Field(min_length=1)
    discount_rate: float = Field(0.03, ge=0, le=1)
    costs: dict
    qalys: dict
    transition_matrices: dict
    productivity_costs: Optional[dict] = None
    friction_cost_params: Optional[dict] = None
    productivity_loss_states: Optional[dict] = None


__all__ = [
    "PSAResultsSchema",
    "ParametersModel",
    "validate_psa_results",
]
