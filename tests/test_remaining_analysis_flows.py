import os
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.bia_model import bia_to_markdown_table, project_bia
from src.dcea_analysis import (
    DCEAnalyzer,
    DCEDataProcessor,
    calculate_analytical_capacity_costs,
    design_dce_study,
    integrate_dce_with_cea,
)
from src.main_analysis import (
    convert_numpy_types,
    generate_literature_informed_dcea_view,
    generate_policy_implications_report,
    perform_dcea_analysis,
    perform_voi_analysis,
    write_results_to_files,
)
from src.threshold_analysis import run_threshold_analysis
from src.validation import PSAResultsSchema, validate_psa_results
from src.value_of_information import (
    ProbabilisticSensitivityAnalysis,
    calculate_evpi,
    calculate_evppi,
    explain_value_of_information_benefits,
    generate_voi_report,
)


@pytest.fixture()
def base_params() -> Dict:
    """Minimal parameters for Markov-based functions."""
    return {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.05, 0.05], [0.0, 0.85, 0.15], [0.0, 0.0, 1.0]],
            "new_treatment": [[0.92, 0.05, 0.03], [0.0, 0.88, 0.12], [0.0, 0.0, 1.0]],
        },
        "cycles": 5,
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
        "friction_cost_params": {
            "friction_period_days": 30,
            "replacement_cost_per_day": 200,
            "absenteeism_rate": 0.02,
        },
        "productivity_loss_states": {"Sick": 5},
    }


def test_validation_threshold_and_reporting(tmp_path, base_params):
    df = pd.DataFrame(
        {"qaly_sc": [1.0, 0.9], "qaly_nt": [1.1, 1.0], "cost_sc": [1000, 950], "cost_nt": [1100, 990]}
    )
    validated = validate_psa_results(df)
    assert isinstance(validated, pd.DataFrame)
    assert PSAResultsSchema.validate(df).shape == (2, 4)

    parameter_ranges = {"new_treatment_cost_multiplier": [0.9, 1.1]}
    thresholds = run_threshold_analysis("Demo", base_params.copy(), parameter_ranges, wtp_threshold=20000)
    assert "new_treatment_cost_multiplier" in thresholds
    assert not thresholds["new_treatment_cost_multiplier"].empty

    report = generate_policy_implications_report(
        {
            "Test": {
                "health_system": {"icer": 1000.0, "incremental_nmb": 5000.0},
                "societal": {"human_capital": {"icer": 800.0, "incremental_nmb": 7000.0}},
            }
        }
    )
    assert "intervention_level_differences" in report

    bia_df = project_bia(
        population_size=1000,
        eligible_prop=0.1,
        uptake_by_year=[0.1, 0.2],
        cost_per_patient=200,
        offset_cost_per_patient=50,
        horizon_years=2,
    )
    md = bia_to_markdown_table(bia_df, currency="NZD", base_year="2024")
    assert "Year" in md and len(md.splitlines()) > 2


def test_value_of_information_pipeline():
    def demo_model(params, intervention_type="standard_care"):
        base_cost = 1000 + params["cost_shift"]
        base_qaly = 1.0 + params["qaly_shift"]
        if intervention_type == "new_treatment":
            base_cost *= 1.05
            base_qaly *= 1.1
        return float(base_cost), float(base_qaly)

    parameters = {
        "cost_shift": {"distribution": "normal", "params": {"mean": 0.0, "std": 10.0}},
        "qaly_shift": {"distribution": "normal", "params": {"mean": 0.0, "std": 0.01}},
    }
    psa = ProbabilisticSensitivityAnalysis(demo_model, parameters, wtp_threshold=20000)
    psa_results = psa.run_psa(n_samples=30)
    evpi = calculate_evpi(psa_results, wtp_threshold=20000)
    assert evpi >= 0

    evppi = calculate_evppi(
        psa_results,
        parameter_group=["cost_shift"],
        all_params=list(parameters.keys()),
        wtp_thresholds=[10000, 20000],
        n_bootstrap=5,
    )
    assert len(evppi) == 2

    voi_report = generate_voi_report(
        psa_results,
        wtp_thresholds=[10000, 20000],
        target_population=5000,
        parameter_names=list(parameters.keys()),
    )
    assert "value_of_information" in voi_report

    explanation = explain_value_of_information_benefits(base_icer=9000, wtp_threshold=20000)
    assert explanation["value_of_information_justification"]


def test_dcea_analysis_pipeline(tmp_path):
    csv_path = tmp_path / "dce_choices.csv"
    choice_rows = [
        {"respondent_id": 1, "choice_task": 1, "alternative": 1, "choice": 1, "cost_per_qaly": 10000, "population_size": 1000, "stakeholder_type": "patient"},
        {"respondent_id": 1, "choice_task": 1, "alternative": 2, "choice": 0, "cost_per_qaly": 15000, "population_size": 800, "stakeholder_type": "patient"},
        {"respondent_id": 2, "choice_task": 1, "alternative": 1, "choice": 0, "cost_per_qaly": 12000, "population_size": 900, "stakeholder_type": "clinician"},
        {"respondent_id": 2, "choice_task": 1, "alternative": 2, "choice": 1, "cost_per_qaly": 13000, "population_size": 950, "stakeholder_type": "clinician"},
    ]
    pd.DataFrame(choice_rows).to_csv(csv_path, index=False)

    processor = DCEDataProcessor()
    loaded = processor.load_choice_data(
        str(csv_path), choice_col="choice", id_col="respondent_id", task_col="choice_task"
    )
    assert not loaded.empty
    processor.define_attributes(
        {
            "cost_per_qaly": {"levels": [10000, 20000], "type": "continuous", "description": "cost"},
            "population_size": {"levels": [800, 1200], "type": "continuous", "description": "size"},
        }
    )
    prepared = processor.prepare_for_modelling()
    assert "cost_per_qaly" in prepared.columns

    analyzer = DCEAnalyzer(processor)
    model_results = analyzer.fit_conditional_logit(
        choice_col="choice", alt_id_col="alternative", attributes=["cost_per_qaly", "population_size"]
    )
    analyzer.model_results["estimated_coefficients"] = {"coef": model_results["estimated_coefficients"]}
    pref_results = analyzer.analyze_stakeholder_preferences(
        attribute_importance=True, willingness_to_pay=True, heterogeneity_analysis=True
    )
    assert "attribute_importance" in pref_results

    study_design = design_dce_study(
        {
            "cost_per_qaly": {"levels": [10000, 20000], "type": "continuous", "description": "cost"},
            "population_size": {"levels": [800, 1200], "type": "continuous", "description": "size"},
        },
        num_choices=3,
        alternatives_per_task=2,
        num_repeats=1,
    )
    assert not study_design.empty

    integrated = integrate_dce_with_cea({"attribute_importance": {"cost_per_qaly": 50}}, {"incremental_nmb": 1000, "is_cost_effective": True})
    assert integrated["integrated_analysis"]

    costs = calculate_analytical_capacity_costs(
        dce_study_size=20, stakeholder_groups=["patients", "clinicians"], country="NZ"
    )
    assert costs["total_cost"] > 0


def test_main_analysis_helpers_and_writing(tmp_path, base_params, monkeypatch):
    import src.main_analysis as ma

    class MiniPSA(ma.ProbabilisticSensitivityAnalysis):
        def run_psa(self, n_samples=10):
            return pd.DataFrame(
                {
                    "qaly_sc": [1.0] * n_samples,
                    "qaly_nt": [1.05] * n_samples,
                    "cost_sc": [1000.0] * n_samples,
                    "cost_nt": [1050.0] * n_samples,
                    "inc_cost": [50.0] * n_samples,
                    "inc_qaly": [0.05] * n_samples,
                }
            )

    monkeypatch.setattr(ma, "ProbabilisticSensitivityAnalysis", MiniPSA)
    voi_report = perform_voi_analysis(base_params)
    assert "value_of_information" in voi_report

    dcea_results = perform_dcea_analysis()
    assert dcea_results

    intervention_results = {
        "Demo": {
            "health_system": {"incremental_nmb": 1000.0, "icer": 5000.0},
            "societal": {"human_capital": {"incremental_nmb": 1500.0, "icer": 4000.0}},
        }
    }
    lit_df = generate_literature_informed_dcea_view(intervention_results)
    assert not lit_df.empty

    policy_report = generate_policy_implications_report(intervention_results)
    assert "recommendations" in policy_report

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    stub_results = {
        "comparative_icer_table": pd.DataFrame([{"icer": 1000, "intervention": "Demo"}]),
        "parameters_table": pd.DataFrame([{"param": "x", "value": 1}]),
        "voi_analysis": {
            "summary_statistics": {"mean_incremental_cost": 1.0, "mean_incremental_qaly": 0.1, "mean_icer": 10.0, "probability_cost_effective": 0.5},
            "value_of_information": {
                "evpi_per_person": 1.0,
                "population_evpi": 10.0,
                "target_population_size": 10,
                "evppi_by_parameter_group": {"EVPPI_base": [0.1]},
                "wtp_thresholds": [50000],
            },
            "methodology_explanation": {"purpose": "demo", "relevance": "demo", "decision_context": "demo"},
        },
        "intervention_results": intervention_results,
    }
    write_results_to_files(stub_results, output_dir=str(out_dir))
    assert any(out_dir.iterdir())

    converted = convert_numpy_types({"a": np.int64(1), "b": np.array([1, 2])})
    assert converted["a"] == 1 and converted["b"] == [1, 2]
