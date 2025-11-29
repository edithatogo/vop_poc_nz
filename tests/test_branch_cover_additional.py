import numpy as np
import pandas as pd
import pytest

import vop_poc_nz.dcea_equity_analysis as equity
from vop_poc_nz.cea_model_core import MarkovModel, _validate_model_parameters
from vop_poc_nz.cluster_analysis import ClusterAnalysis
from vop_poc_nz.dsa_analysis import (
    perform_comprehensive_two_way_dsa,
    perform_three_way_dsa,
)
from vop_poc_nz.threshold_analysis import run_threshold_analysis
from vop_poc_nz.value_of_information import ProbabilisticSensitivityAnalysis


def minimal_params():
    return {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.05, 0.05], [0.0, 0.85, 0.15], [0.0, 0.0, 1.0]],
            "new_treatment": [[0.92, 0.05, 0.03], [0.0, 0.88, 0.12], [0.0, 0.0, 1.0]],
        },
        "cycles": 2,
        "initial_population": [1000, 0, 0],
        "discount_rate": 0.03,
        "costs": {
            "health_system": {
                "standard_care": [0, 500, 0],
                "new_treatment": [100, 450, 0],
            },
            "societal": {
                "standard_care": [0, 800, 0],
                "new_treatment": [50, 500, 0],
            },
        },
        "qalys": {"standard_care": [1.0, 0.7, 0.0], "new_treatment": [1.0, 0.8, 0.0]},
        "productivity_costs": {
            "human_capital": {
                "standard_care": [0, 200, 0],
                "new_treatment": [0, 120, 0],
            }
        },
        "productivity_loss_states": {"Sick": 5},
        "friction_cost_params": {
            "friction_period_days": 30,
            "replacement_cost_per_day": 200,
            "absenteeism_rate": 0.02,
        },
    }


def test_markov_and_validation_errors():
    # transition matrix dimension and row-sum mismatch
    with pytest.raises(ValueError):
        MarkovModel(["A"], [[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        MarkovModel(["A", "B"], [[0.5, 0.6], [0.4, 0.5]])
    model = MarkovModel(["A", "B"], [[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ValueError):
        model.run(cycles=1, initial_population=[1], costs=[1, 1], qalys=[1, 1])
    with pytest.raises(ValueError):
        model.run(cycles=1, initial_population=[1, 1], costs=[1], qalys=[1, 1])
    with pytest.raises(ValueError):
        model.run(cycles=1, initial_population=[1, 1], costs=[1, 1], qalys=[1])

    with pytest.raises(ValueError):
        _validate_model_parameters({})
    with pytest.raises(ValueError):
        _validate_model_parameters(
            {
                "states": [],
                "transition_matrices": {},
                "cycles": 1,
                "initial_population": [],
                "costs": {},
                "qalys": {},
            }
        )
    with pytest.raises(ValueError):
        _validate_model_parameters(
            {
                "states": ["A"],
                "transition_matrices": {"standard_care": [[1]], "new_treatment": [[1]]},
                "cycles": 1,
                "initial_population": [1],
                "costs": {"health_system": {}, "societal": {}},
                "qalys": {"standard_care": [1]},
            }
        )


def test_cluster_branches_cover():
    rng = np.random.default_rng(1)
    results = {
        "HPV Vaccination": {
            "inc_cost": rng.normal(0, 1, size=30),
            "inc_qaly": rng.normal(0.1, 0.02, size=30),
        }
    }
    analyzer = ClusterAnalysis(results, ["HPV Vaccination"])
    res = analyzer.prepare_clustering_data("HPV Vaccination", n_simulations=10)
    assert res.shape[0] == 10

    smoking_results = {
        "Smoking Cessation": {
            "inc_cost": rng.normal(0, 1, size=20),
            "inc_qaly": rng.normal(0.1, 0.02, size=20),
        }
    }
    smoking_analyzer = ClusterAnalysis(smoking_results, ["Smoking Cessation"])
    res_smoking = smoking_analyzer.prepare_clustering_data(
        "Smoking Cessation", n_simulations=5
    )
    assert res_smoking.shape[0] == 5


def test_dsa_branch_variants():
    base = minimal_params()
    interventions = {
        "HPV Vaccine": base,
        "Smoking Cessation": base,
        "Hepatitis C": base,
        "Obesity Program": base,
        "Housing Upgrade": base,
    }
    two_way = perform_comprehensive_two_way_dsa(
        interventions, wtp_threshold=10000, n_points=1
    )
    assert two_way
    three_way = perform_three_way_dsa(interventions, wtp_threshold=10000, n_points=1)
    assert three_way


def test_threshold_qaly_branch():
    params = minimal_params()
    ranges = {"new_treatment_qaly_multiplier": [0.9, 1.1]}
    out = run_threshold_analysis("Demo", params, ranges, wtp_threshold=20000)
    assert "new_treatment_qaly_multiplier" in out


def test_value_of_information_ceac():
    psa_results = pd.DataFrame(
        {
            "qaly_sc": [1.0, 0.9],
            "qaly_nt": [1.2, 1.1],
            "cost_sc": [1000, 950],
            "cost_nt": [1100, 980],
            "inc_cost": [100, 30],
            "inc_qaly": [0.2, 0.2],
        }
    )
    psa = ProbabilisticSensitivityAnalysis(
        model_func=lambda params, intervention_type=None: (0, 0),
        parameters={
            "dummy": {"distribution": "normal", "params": {"mean": 0, "std": 1}}
        },
        wtp_threshold=20000,
    )
    ceac = psa.calculate_ceac(psa_results, wtp_values=[10000, 20000])
    assert not ceac.empty

    with pytest.raises(ValueError):
        psa.parameters = {"bad": {"distribution": "unknown", "params": {}}}
        psa.sample_parameters(n_samples=1)
    psa.parameters = {
        "dummy": {"distribution": "normal", "params": {"mean": 0, "std": 1}}
    }

    from vop_poc_nz.value_of_information import (
        calculate_evppi,
        explain_value_of_information_benefits,
        generate_voi_report,
    )

    evppi_missing = calculate_evppi(
        psa_results,
        parameter_group=["absent"],
        all_params=["another"],
        wtp_thresholds=None,
        n_bootstrap=1,
    )
    assert all(v == 0.0 for v in evppi_missing)

    explanation = explain_value_of_information_benefits(
        base_icer=60000, wtp_threshold=50000
    )
    assert explanation["value_of_information_justification"]

    voi = generate_voi_report(
        psa_results, wtp_thresholds=None, target_population=1000, parameter_names=None
    )
    assert "value_of_information" in voi


def test_equity_metrics_branches():
    gini = equity.calculate_gini([-1, 0, 1])
    assert gini >= 0
    atk = equity.calculate_atkinson_index([1, 2, 3], epsilon=1)
    assert atk >= 0

    text = equity.generate_dcea_results_table(
        {
            "distribution_of_net_health_benefits": {},
            "total_health_gain": "N/A",
            "variance_of_net_health_benefits": "N/A",
            "gini_coefficient": 0.0,
            "atkinson_index": 0.0,
            "atkinson_epsilon": 1.0,
        },
        "Demo",
    )
    assert isinstance(text, pd.DataFrame)
