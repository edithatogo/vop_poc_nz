import pandas as pd

from src.dcea_analysis import (
    calculate_analytical_capacity_costs,
    integrate_dce_with_cea,
)
from src.dcea_equity_analysis import run_dcea
from src.reporting import generate_comprehensive_report
from src.value_of_information import calculate_evppi

MINIMAL_PARAMS = {
    "states": ["Healthy", "Sick"],
    "transition_matrices": {
        "standard_care": [[0.9, 0.1], [0.0, 1.0]],
        "new_treatment": [[0.95, 0.05], [0.0, 1.0]],
    },
    "cycles": 2,
    "initial_population": [100, 0],
    "costs": {
        "health_system": {
            "standard_care": [0, 100],
            "new_treatment": [50, 80],
        },
        "societal": {
            "standard_care": [0, 150],
            "new_treatment": [60, 90],
        },
    },
    "qalys": {
        "standard_care": [1.0, 0.7],
        "new_treatment": [1.0, 0.8],
    },
    "productivity_costs": {
        "human_capital": {
            "standard_care": [0, 50],
            "new_treatment": [0, 30],
        }
    },
    "discount_rate": 0.03,
    "friction_cost_params": {
        "friction_period_days": 100,
        "replacement_cost_per_day": 200,
        "absenteeism_rate": 0.05,
    },
    "productivity_loss_states": {"Sick": 10},
}


def test_dcea_integration_minimal():
    # Build minimal DCE results structure expected by integrator
    dce_results = {
        "attribute_importance": {"cost_per_qaly": 1.0, "population_size": 0.5}
    }
    integrated = integrate_dce_with_cea(
        dce_results, {"incremental_nmb": 0, "is_cost_effective": True}
    )
    assert "policy_recommendations" in integrated


def test_dcea_equity_run_minimal():
    subgroup_data = {
        "High_SES": {"incremental_nmb": 1000},
        "Low_SES": {"incremental_nmb": 800},
    }
    dcea = run_dcea(subgroup_data)
    assert "gini_coefficient" in dcea


def test_evppi_minimal():
    df = pd.DataFrame(
        {
            "param": [0.1, 0.2, 0.3],
            "qaly_sc": [1.0, 1.0, 1.0],
            "qaly_nt": [1.1, 1.2, 1.05],
            "cost_sc": [1000, 1100, 900],
            "cost_nt": [1200, 1150, 950],
        }
    )
    evppi = calculate_evppi(
        df,
        parameter_group=["param"],
        all_params=list(df.columns),
        wtp_thresholds=[10000, 20000],
    )
    assert isinstance(evppi, list)


def test_reporting_minimal_dcea():
    report = generate_comprehensive_report("Test", MINIMAL_PARAMS)
    assert "Comprehensive CEA Report" in report


def test_analytical_capacity_costs():
    cost_info = calculate_analytical_capacity_costs(
        dce_study_size=1000, stakeholder_groups=["clinicians", "patients"]
    )
    assert "total_cost" in cost_info
