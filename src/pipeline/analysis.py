"""
Core Analysis Pipeline Module.

This module contains the logic for running the health economic analysis,
including CEA, DCEA, VOI, and DSA. It separates the "math" from the "reporting".
"""

import copy
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from ..bia_model import project_bia
from ..cea_model_core import (
    create_parameters_table,
    generate_comparative_icer_table,
    run_cea,
)
from ..dcea_equity_analysis import run_dcea
from ..discordance_analysis import calculate_decision_discordance
from ..reporting import generate_comprehensive_report
from ..threshold_analysis import run_threshold_analysis
from ..value_of_information import (
    ProbabilisticSensitivityAnalysis,
)


def load_parameters(filepath: str = "src/parameters.yaml") -> Dict:
    """Load parameters from a YAML file."""
    # Path relative to project root (assuming run from root)
    if os.path.exists(filepath):
        with open(filepath) as f:
            return yaml.safe_load(f)

    # Fallback for running from src/pipeline
    project_root = Path(__file__).parent.parent.parent
    full_path = project_root / filepath
    if full_path.exists():
        with open(full_path) as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find parameters file at {filepath} or {full_path}"
    )


def perform_voi_analysis(params: Dict) -> Dict:
    """Placeholder for VOI analysis."""
    return {
        "value_of_information": {
            "evppi_by_parameter_group": {},
            "wtp_thresholds": list(np.linspace(0, 100000, 21)),
        }
    }


def perform_dsa_analysis(interventions: Dict) -> Dict:
    """Placeholder for DSA analysis."""
    return {"1_way": {}, "2_way": {}, "3_way": {}}


def calculate_analytical_capacity_costs(
    dce_study_size: int, stakeholder_groups: list, country: str
) -> Dict:
    """Placeholder for capacity cost calculation."""
    return {"total_cost": 0}


def generate_cheers_report() -> str:
    """Placeholder for CHEERS compliance report."""
    return "CHEERS compliance report placeholder"


def run_analysis_pipeline() -> Dict:
    """
    Run the complete health economic analysis pipeline.

    Performs CEA, DCEA, VOI, DSA, BIA for all selected interventions.

    Returns:
        Dict with all analysis results
    """
    # Load all parameters
    all_params = load_parameters()

    # Select interventions to analyze
    selected_interventions = {
        "HPV Vaccination": copy.deepcopy(all_params["hpv_vaccination"]),
        "Smoking Cessation": copy.deepcopy(all_params["smoking_cessation"]),
        "Hepatitis C Therapy": copy.deepcopy(all_params["hepatitis_c_therapy"]),
        "Childhood Obesity Prevention": copy.deepcopy(
            all_params["childhood_obesity_prevention"]
        ),
        "Housing Insulation": copy.deepcopy(all_params["housing_insulation"]),
    }

    all_results = {}
    comparison_tables = []
    parameters_tables = []

    for name, params in selected_interventions.items():
        print(f"\nAnalyzing {name}...")

        # 1. Health System Perspective
        hs_results = run_cea(params, perspective="health_system")
        all_results[name] = {"health_system": hs_results, "societal": {}}

        # 2. Societal Perspective (Human Capital & Friction Cost)
        for method in ["human_capital", "friction_cost"]:
            print(f"  ... with {method} method")
            s_results = run_cea(
                params,
                perspective="societal",
                productivity_cost_method=method,
            )
            all_results[name]["societal"][method] = s_results

            comp_table = generate_comparative_icer_table(
                hs_results, s_results, f"{name} ({method})"
            )
            comparison_tables.append(comp_table)

            # DCEA (Equity Analysis)
            if s_results.get("subgroup_results"):
                print(f"  Performing Equity Analysis for {name} ({method})...")
                equity_weights = {
                    "Low_SES": 1.5,
                    "High_SES": 1.0,
                    "Māori": 1.5,
                    "Non-Māori": 1.0,
                }
                dcea_res = run_dcea(
                    s_results["subgroup_results"],
                    epsilon=0.5,
                    equity_weights=equity_weights,
                )
                all_results[name]["societal"][method]["dcea_equity_analysis"] = dcea_res
                print(
                    f"  Equity-Weighted Net Benefit: ${dcea_res.get('weighted_total_health_gain', 0):,.0f}"
                )

        # 3. Discordance Analysis
        all_results[name]["discordance"] = calculate_decision_discordance(name, params)

        # 4. Parameters Table
        sources = {
            "transition_Healthy_Sick_standard": "Model assumption",
            "cost_hs_Sick_standard": "Based on New Zealand health system data",
            "cost_societal_Sick_standard": "Productivity loss estimates from literature",
            "cycles": "Lifetime analysis as per manuscript",
            "qaly_Healthy_standard": "Standard utility values from literature",
        }
        param_table = create_parameters_table(params, sources)
        param_table["intervention"] = name
        parameters_tables.append(param_table)

    full_comparison = pd.concat(comparison_tables, ignore_index=True)
    full_parameters = pd.concat(parameters_tables, ignore_index=True)

    # 5. Value of Information
    print("\nRunning Value of Information Analysis...")
    voi_results = perform_voi_analysis(selected_interventions["HPV Vaccination"])

    # 6. CHEERS Report
    cheers_report = generate_cheers_report()

    # 7. Threshold Analysis
    print("\nPerforming Threshold Analysis...")
    threshold_results = {}
    for name, params in selected_interventions.items():
        parameter_ranges = {
            "cost_hs_new_treatment_state_0": np.linspace(
                params["costs"]["health_system"]["new_treatment"][0] * 0.5,
                params["costs"]["health_system"]["new_treatment"][0] * 1.5,
                20,
            )
        }
        threshold_results[name] = run_threshold_analysis(name, params, parameter_ranges)

    # 8. Probabilistic Sensitivity Analysis (PSA)
    print("\nPerforming Probabilistic Sensitivity Analysis (PSA)...")
    probabilistic_results = {}
    for name, params in selected_interventions.items():
        psa_distributions = {
            "cost_new_treatment_multiplier": {
                "distribution": "normal",
                "params": {"mean": 1.0, "std": 0.1},
            },
            "qaly_new_treatment_multiplier": {
                "distribution": "normal",
                "params": {"mean": 1.0, "std": 0.05},
            },
            "transition_sick_dead_rate_new_treatment": {
                "distribution": "beta",
                "params": {"alpha": 2, "beta": 18},
            },
        }

        def psa_run_cea_wrapper(sampled_params, intervention_type, base_params=params):
            temp_params = copy.deepcopy(base_params)
            temp_params["costs"]["health_system"]["new_treatment"][0] *= sampled_params[
                "cost_new_treatment_multiplier"
            ]
            temp_params["costs"]["societal"]["new_treatment"][0] *= sampled_params[
                "cost_new_treatment_multiplier"
            ]

            if (
                "qalys" in temp_params
                and len(temp_params["qalys"]["new_treatment"]) > 1
            ):
                temp_params["qalys"]["new_treatment"][1] *= sampled_params[
                    "qaly_new_treatment_multiplier"
                ]

            if (
                "transition_matrices" in temp_params
                and len(temp_params["transition_matrices"]["new_treatment"]) > 1
            ):
                temp_params["transition_matrices"]["new_treatment"][1][2] = (
                    sampled_params["transition_sick_dead_rate_new_treatment"]
                )
                row_sum = sum(temp_params["transition_matrices"]["new_treatment"][1])
                if row_sum != 1.0:
                    temp_params["transition_matrices"]["new_treatment"][1][1] = (
                        1.0
                        - temp_params["transition_matrices"]["new_treatment"][1][0]
                        - temp_params["transition_matrices"]["new_treatment"][1][2]
                    )

            cea_results_dict = run_cea(
                temp_params, perspective="societal", wtp_threshold=50000
            )

            if intervention_type == "standard_care":
                return (
                    cea_results_dict["cost_standard_care"],
                    cea_results_dict["qalys_standard_care"],
                )
            elif intervention_type == "new_treatment":
                return (
                    cea_results_dict["cost_new_treatment"],
                    cea_results_dict["qalys_new_treatment"],
                )
            else:
                raise ValueError("Invalid intervention_type")

        psa = ProbabilisticSensitivityAnalysis(
            psa_run_cea_wrapper, psa_distributions, wtp_threshold=50000
        )
        probabilistic_results[name] = psa.run_psa(n_samples=1000)

    # 9. Budget Impact Analysis (BIA)
    print("\nPerforming Budget Impact Analysis (BIA)...")
    bia_results = {}
    for name, params in selected_interventions.items():
        # Use intervention-specific population parameters
        bia_pop = params.get("bia_population", {})
        total_pop = bia_pop.get("total_population", 100000)  # Fallback to 100k
        eligible_prop = bia_pop.get("eligible_proportion", 0.1)  # Fallback to 10%

        bia_params = {
            "population_size": total_pop,
            "eligible_prop": eligible_prop,
            "uptake_by_year": [0.1, 0.2, 0.3, 0.4, 0.5],
            "cost_per_patient": params["costs"]["health_system"]["new_treatment"][0],
            "offset_cost_per_patient": params["costs"]["health_system"][
                "standard_care"
            ][0],
            "discount_rate": params.get(
                "discount_rate", 0.03
            ),  # Use parameter discount rate
        }
        bia_results[name] = project_bia(**bia_params)
        print(
            f"  {name}: Population={total_pop:,}, Eligible={eligible_prop:.0%} ({total_pop *eligible_prop:,.0f} people)"
        )

    # 10. Comprehensive Reports
    print("\nGenerating Comprehensive Reports...")
    reports = {}
    for name, params in selected_interventions.items():
        reports[name] = generate_comprehensive_report(name, params)

    # 11. DSA
    dsa_results = perform_dsa_analysis(selected_interventions)

    # 12. Cluster Analysis (Placeholder)
    cluster_results = {}

    return {
        "intervention_results": all_results,
        "comparative_icer_table": full_comparison,
        "parameters_table": full_parameters,
        "voi_analysis": voi_results,
        "cheers_compliance": cheers_report,
        "threshold_analysis": threshold_results,
        "probabilistic_results": probabilistic_results,
        "bia_results": bia_results,
        "reports": reports,
        "dsa_analysis": dsa_results,
        "cluster_analysis": cluster_results,
        "dcea_equity_analysis": {
            name: res["societal"].get("human_capital", {}).get("dcea_equity_analysis")
            for name, res in all_results.items()
            if res.get("societal", {})
            .get("human_capital", {})
            .get("dcea_equity_analysis")
        },
        "selected_interventions": selected_interventions,  # Needed for plotting
    }
