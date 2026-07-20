import numpy as np
import pandas as pd

from vop_poc_nz.cea_model_core import run_cea
from vop_poc_nz.cluster_analysis import ClusterAnalysis
from vop_poc_nz.discordance_analysis import calculate_decision_discordance
from vop_poc_nz.dsa_analysis import perform_one_way_dsa
from vop_poc_nz.reporting import generate_comprehensive_report
from vop_poc_nz.threshold_analysis import run_threshold_analysis
from vop_poc_nz.value_of_information import calculate_evpi

MINIMAL_PARAMS = {
    "states": ["Healthy", "Sick"],
    "transition_matrices": {
        "standard_care": [[0.9, 0.1], [0.0, 1.0]],
        "new_treatment": [[0.95, 0.05], [0.0, 1.0]],
    },
    "cycles": 3,
    "initial_population": [1000, 0],
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


def test_run_cea_minimal():
    result = run_cea(MINIMAL_PARAMS, perspective="societal", wtp_threshold=50000)
    assert "incremental_cost" in result
    assert "incremental_qalys" in result


def test_evpi_minimal():
    df = pd.DataFrame(
        {
            "qaly_sc": [1.0, 1.0, 1.0],
            "qaly_nt": [1.1, 1.2, 1.05],
            "cost_sc": [1000, 1100, 900],
            "cost_nt": [1200, 1150, 950],
        }
    )
    evpi = calculate_evpi(df, wtp_threshold=50000)
    assert isinstance(evpi, float)


def test_one_way_dsa_minimal():
    results = perform_one_way_dsa(
        {"Test": MINIMAL_PARAMS}, wtp_threshold=20000, n_points=3
    )
    assert "Test" in results
    assert "dsa_results" in results["Test"]


def test_reporting_minimal():
    report = generate_comprehensive_report("Test", MINIMAL_PARAMS)
    assert "Comprehensive CEA Report" in report


def test_threshold_analysis_minimal():
    param_ranges = {"cost_multiplier": np.linspace(0.8, 1.2, 3)}
    thres = run_threshold_analysis("Test", MINIMAL_PARAMS, param_ranges)
    assert "cost_multiplier" in thres


def test_cluster_analysis_minimal():
    # Build minimal probabilistic results for one intervention
    inc_cost = np.random.normal(0, 1000, 50)
    inc_qaly = np.random.normal(0.1, 0.05, 50)
    probabilistic_results = {"Test": {"inc_cost": inc_cost, "inc_qaly": inc_qaly}}
    ca = ClusterAnalysis(probabilistic_results, ["Test"])
    res = ca.prepare_clustering_data("Test", n_simulations=20)
    assert res.shape[1] >= 9
    clustering = ca.perform_clustering("Test", n_clusters_range=range(2, 3))
    assert "cluster_labels" in clustering


def test_discordance_analysis_minimal():
    result = calculate_decision_discordance("Test", MINIMAL_PARAMS)
    assert "discordant" in result
