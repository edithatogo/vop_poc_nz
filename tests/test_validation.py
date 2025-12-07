import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.validation import ParametersModel, validate_psa_results


def test_validate_psa_results_passes():
    df = pd.DataFrame(
        {
            "qaly_sc": [1.0, 0.9],
            "qaly_nt": [1.1, 1.0],
            "cost_sc": [1000, 900],
            "cost_nt": [1100, 950],
        }
    )
    validated = validate_psa_results(df)
    assert list(validated.columns) == list(df.columns)


def test_validate_psa_results_fails_missing_column():
    df = pd.DataFrame(
        {
            "qaly_sc": [1.0],
            "cost_sc": [1000],
            "cost_nt": [1100],
        }
    )
    with pytest.raises(SchemaError):
        validate_psa_results(df)


def test_parameters_model_basic():
    params = ParametersModel(
        states=["Healthy", "Sick"],
        cycles=2,
        initial_population=[100, 0],
        discount_rate=0.03,
        costs={},
        qalys={},
        transition_matrices={},
    )
    assert params.cycles == 2
