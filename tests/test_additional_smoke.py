import pandas as pd

from src.cea_model_core import run_cea
from src.cluster_analysis import ClusterAnalysis
from src.dsa_analysis import perform_one_way_dsa
from src.threshold_analysis import run_threshold_analysis
from src.value_of_information import (
    calculate_evpi,
    calculate_evppi,
    calculate_population_evpi,
    explain_value_of_information_benefits,
)
from src.visualizations import plot_cumulative_nmb, plot_efficiency_frontier


MINIMAL_PARAMS = {
    "states": ["Healthy", "Dead"],
    "transition_matrices": {
        "standard_care": [[0.95, 0.05], [0.0, 1.0]],
        "new_treatment": [[0.97, 0.03], [0.0, 1.0]],
    },
    "cycles": 2,
    "initial_population": [1000, 0],
    "costs": {
        "health_system": {
            "standard_care": [0, 100],
            "new_treatment": [20, 120],
        },
        "societal": {
            "standard_care": [0, 150],
            "new_treatment": [30, 140],
        },
    },
    "qalys": {
        "standard_care": [1.0, 0.7],
        "new_treatment": [1.0, 0.75],
    },
    "productivity_costs": {
        "human_capital": {
            "standard_care": [0, 50],
            "new_treatment": [0, 30],
        }
    },
    "discount_rate": 0.03,
    "friction_cost_params": {
        "friction_period_days": 30,
        "replacement_cost_per_day": 200,
        "absenteeism_rate": 0.05,
    },
    "productivity_loss_states": {"Dead": 0},
}


def test_threshold_and_dsa_smoke():
    # Threshold analysis should return results for provided parameter
    param_ranges = {"cost_multiplier": [0.8, 1.2]}
    threshold_results = run_threshold_analysis("Test", MINIMAL_PARAMS, param_ranges)
    assert "cost_multiplier" in threshold_results

    # One-way DSA should process without error
    dsa_results = perform_one_way_dsa({"Test": MINIMAL_PARAMS}, wtp_threshold=20000, n_points=3)
    assert "Test" in dsa_results


def test_cluster_and_voi_smoke(tmp_path):
    # Cluster analysis on simple simulated data
    inc_cost = [0.0, 100.0, -50.0, 20.0]
    inc_qaly = [0.1, 0.2, 0.05, 0.15]
    analyzer = ClusterAnalysis({"Test": {"inc_cost": inc_cost, "inc_qaly": inc_qaly}}, ["Test"])
    data = analyzer.prepare_clustering_data("Test", n_simulations=4)
    assert data.shape[0] == 4

    # VOI utilities
    psa_df = pd.DataFrame(
        {
            "qaly_sc": [1.0, 0.9],
            "qaly_nt": [1.1, 1.0],
            "cost_sc": [1000, 900],
            "cost_nt": [1100, 950],
        }
    )
    evpi = calculate_evpi(psa_df, wtp_threshold=50000)
    assert evpi >= 0
    evppi = calculate_evppi(psa_df, ["cost_nt"], ["cost_nt", "cost_sc", "qaly_sc", "qaly_nt"], wtp_thresholds=[50000])
    assert len(evppi) == 1
    population_evpi = calculate_population_evpi(evpi, target_population_size=1000)
    assert population_evpi >= evpi
    explain = explain_value_of_information_benefits(base_icer=20000, wtp_threshold=50000)
    assert explain.get("base_case_info") is not None


def test_plotting_frontiers(tmp_path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir()

    # Efficiency frontier plot
    costs = [10000, 12000, 15000]
    qalys = [10.0, 11.0, 12.5]
    strategies = ["A", "B", "C"]
    plot_efficiency_frontier(costs, qalys, strategies, output_dir=str(out_dir))

    # Cumulative NMB plot
    plot_cumulative_nmb([1000.0, -500.0, 200.0], output_dir=str(out_dir))

    assert any(out_dir.iterdir())


def test_run_cea_minimal():
    result = run_cea(MINIMAL_PARAMS, perspective="societal", wtp_threshold=50000)
    assert "incremental_cost" in result
