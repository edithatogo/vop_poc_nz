import pandas as pd

from src.bia_model import bia_to_markdown_table
from src.cea_model_core import run_cea
from src.dcea_equity_analysis import run_dcea
from src.main_analysis import generate_cheers_report
from src.reporting import generate_comprehensive_report, generate_dcea_results_table
from src.value_of_information import ProbabilisticSensitivityAnalysis


def minimal_params():
    return {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.05, 0.05], [0.0, 0.85, 0.15], [0.0, 0.0, 1.0]],
            "new_treatment": [[0.92, 0.05, 0.03], [0.0, 0.88, 0.12], [0.0, 0.0, 1.0]],
        },
        "cycles": 3,
        "initial_population": [1000, 0, 0],
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
        "subgroups": {
            "Group1": {"initial_population": [500, 0, 0]},
            "Group2": {"initial_population": [500, 0, 0]},
        },
    }


def test_reporting_and_dcea_table(tmp_path):
    params = minimal_params()
    report = generate_comprehensive_report("Demo", params, wtp_threshold=20000)
    assert "Comprehensive CEA Report" in report

    dcea_results = {
        "distribution_of_net_health_benefits": {"Q1": 100, "Q2": 150},
        "total_health_gain": 250,
        "variance_of_net_health_benefits": 25,
        "gini_coefficient": 0.1,
        "atkinson_index": 0.05,
        "atkinson_epsilon": 0.5,
    }
    table = generate_dcea_results_table(dcea_results)
    assert isinstance(table, str) and "Net Health Benefit" in table

    # run_dcea aggregates subgroup results
    dcea_summary = run_dcea(
        {
            "Q1": {"incremental_nmb": 100},
            "Q2": {"incremental_nmb": 150},
        }
    )
    assert "variance_of_net_health_benefits" in dcea_summary

    # bia_to_markdown_table covers to_int_safe error branch
    df_bad = pd.DataFrame(
        {
            "year": ["bad"],
            "treated": ["n/a"],
            "gross_cost": [1.0],
            "offsets": [0.0],
            "net_cost": [1.0],
        }
    )
    md = bia_to_markdown_table(df_bad)
    assert "| 0 |" in md

    empty = generate_dcea_results_table({})
    assert empty == "DCEA results are not available.\n"

    cheers = generate_cheers_report()
    assert (
        cheers["cheers_2022_compliance"]["met_items"]
        == cheers["cheers_2022_compliance"]["total_items"]
    )


def test_run_cea_subgroups_and_friction():
    params = minimal_params()
    results = run_cea(
        params, perspective="societal", productivity_cost_method="friction_cost"
    )
    assert results["subgroup_results"] is not None


def test_psa_sampling_distributions():
    def model(params, intervention_type="standard_care"):
        base = 100 + params["shift"]
        return (
            (base, base / 50)
            if intervention_type == "standard_care"
            else (base * 1.1, base / 45)
        )

    parameters = {
        "shift": {"distribution": "gamma", "params": {"shape": 2.0, "scale": 5.0}},
        "prob": {"distribution": "beta", "params": {"alpha": 2.0, "beta": 2.0}},
        "norm": {"distribution": "normal", "params": {"mean": 0.0, "std": 1.0}},
        "span": {"distribution": "uniform", "params": {"low": 0.0, "high": 1.0}},
    }
    psa = ProbabilisticSensitivityAnalysis(model, parameters, wtp_threshold=20000)
    samples = psa.sample_parameters(n_samples=5)
    assert len(samples) == 5
    psa_results = psa.run_psa(n_samples=5)
    assert not psa_results.empty
