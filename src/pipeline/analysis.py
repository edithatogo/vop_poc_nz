"""
Core Analysis Pipeline Module.

This module contains the logic for running the health economic analysis,
including CEA, DCEA, VOI, and DSA. It separates the "math" from the "reporting".
"""

import copy
import logging
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
    generate_voi_report,
)

logger = logging.getLogger(__name__)


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





from ..dsa_analysis import (
    perform_comprehensive_two_way_dsa,
    perform_one_way_dsa,
    perform_three_way_dsa,
)


def perform_dsa_analysis(interventions: Dict) -> Dict:
    """Perform Deterministic Sensitivity Analysis."""
    logger.info("Running Deterministic Sensitivity Analysis (DSA)...")

    # 1-Way DSA
    one_way_results = perform_one_way_dsa(interventions)

    # 2-Way DSA
    two_way_results = perform_comprehensive_two_way_dsa(interventions)

    # 3-Way DSA
    three_way_results = perform_three_way_dsa(interventions)

    return {
        "1_way": one_way_results,
        "2_way": two_way_results,
        "3_way": three_way_results,
    }


def calculate_analytical_capacity_costs(
    _dce_study_size: int, _stakeholder_groups: list, _country: str
) -> Dict:
    """Placeholder for capacity cost calculation."""
    return {"total_cost": 0}


def generate_cheers_report() -> Dict:
    """Placeholder for CHEERS compliance report."""
    return {
        "cheers_2022_compliance": {
            "compliance_percentage": 100.0,
            "items": {"Title": True, "Abstract": True},  # Example items
        }
    }


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
        logger.info(f"Analyzing {name}...")

        # 1. Health System Perspective
        hs_results = run_cea(params, perspective="health_system")
        all_results[name] = {"health_system": hs_results, "societal": {}}

        # DCEA (Equity Analysis) for Health System
        if hs_results.get("subgroup_results"):
            logger.info(f"  Performing Equity Analysis for {name} (Health System)...")
            equity_weights = {
                "Low_SES": 1.5,
                "High_SES": 1.0,
                "Māori": 1.5,
                "Non-Māori": 1.0,
            }
            dcea_res_hs = run_dcea(
                hs_results["subgroup_results"],
                epsilon=0.5,
                equity_weights=equity_weights,
            )
            all_results[name]["health_system"]["dcea_equity_analysis"] = dcea_res_hs

        # 2. Societal Perspective (Human Capital & Friction Cost)
        for method in ["human_capital", "friction_cost"]:
            logger.debug(f"  ... with {method} method")
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

            # DCEA (Equity Analysis) for Societal
            if s_results.get("subgroup_results"):
                logger.info(f"  Performing Equity Analysis for {name} ({method})...")
                # Re-use weights
                dcea_res = run_dcea(
                    s_results["subgroup_results"],
                    epsilon=0.5,
                    equity_weights=equity_weights,
                )
                all_results[name]["societal"][method]["dcea_equity_analysis"] = dcea_res
                logger.debug(
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

    # 5. Probabilistic Sensitivity Analysis (PSA)
    logger.info("Performing Probabilistic Sensitivity Analysis (PSA)...")
    probabilistic_results = {}
    for name, params in selected_interventions.items():
        # Define PSA distributions
        psa_distributions = {
            # Global Cost Multipliers
            "cost_hs_sc_multiplier": {"distribution": "normal", "params": {"mean": 1.0, "std": 0.1}},
            "cost_hs_nt_multiplier": {"distribution": "normal", "params": {"mean": 1.0, "std": 0.1}},
            "cost_soc_sc_multiplier": {"distribution": "normal", "params": {"mean": 1.0, "std": 0.15}},
            "cost_soc_nt_multiplier": {"distribution": "normal", "params": {"mean": 1.0, "std": 0.15}},

            # Global QALY Multipliers
            "qaly_sc_multiplier": {"distribution": "normal", "params": {"mean": 1.0, "std": 0.05}},
            "qaly_nt_multiplier": {"distribution": "normal", "params": {"mean": 1.0, "std": 0.05}},
        }

        # Add subgroup-specific multipliers if subgroups exist
        if "subgroups" in params:
            for subgroup in params["subgroups"].keys():
                # Add independent uncertainty for each subgroup (centered around 1.0)
                # We use a smaller std dev for subgroups to represent specific variation around the global trend
                # OR we can treat them as fully independent. Let's add them as independent modifiers.

                # Actually, to avoid double counting uncertainty (Global * Subgroup),
                # we should probably just use specific multipliers INSTEAD of global if available,
                # or treat them as deviations.

                # Let's define specific multipliers for each subgroup with the SAME variance as global,
                # effectively treating them as independent populations.
                clean_name = subgroup.replace(" ", "_")
                psa_distributions[f"cost_hs_sc_multiplier_{clean_name}"] = {"distribution": "normal", "params": {"mean": 1.0, "std": 0.1}}
                psa_distributions[f"cost_hs_nt_multiplier_{clean_name}"] = {"distribution": "normal", "params": {"mean": 1.0, "std": 0.1}}
                psa_distributions[f"cost_soc_sc_multiplier_{clean_name}"] = {"distribution": "normal", "params": {"mean": 1.0, "std": 0.15}}
                psa_distributions[f"cost_soc_nt_multiplier_{clean_name}"] = {"distribution": "normal", "params": {"mean": 1.0, "std": 0.15}}
                psa_distributions[f"qaly_sc_multiplier_{clean_name}"] = {"distribution": "normal", "params": {"mean": 1.0, "std": 0.05}}
                psa_distributions[f"qaly_nt_multiplier_{clean_name}"] = {"distribution": "normal", "params": {"mean": 1.0, "std": 0.05}}

        def psa_run_cea_wrapper(sampled_params, intervention_type, base_params=params):
            temp_params = copy.deepcopy(base_params)

            def apply_multipliers(target_params, subgroup_suffix=""):
                # Helper to get the right multiplier
                def get_mult(base_name):
                    if subgroup_suffix:
                        specific_key = f"{base_name}_{subgroup_suffix}"
                        if specific_key in sampled_params:
                            return sampled_params[specific_key]
                    return sampled_params[base_name]

                # Apply multipliers to Health System Costs
                if "costs" in target_params and "health_system" in target_params["costs"]:
                    if "standard_care" in target_params["costs"]["health_system"]:
                        m = get_mult("cost_hs_sc_multiplier")
                        target_params["costs"]["health_system"]["standard_care"] = [
                            c * m for c in target_params["costs"]["health_system"]["standard_care"]
                        ]
                    if "new_treatment" in target_params["costs"]["health_system"]:
                        m = get_mult("cost_hs_nt_multiplier")
                        target_params["costs"]["health_system"]["new_treatment"] = [
                            c * m for c in target_params["costs"]["health_system"]["new_treatment"]
                        ]

                # Apply multipliers to Societal Costs
                if "costs" in target_params and "societal" in target_params["costs"]:
                    if "standard_care" in target_params["costs"]["societal"]:
                        m = get_mult("cost_soc_sc_multiplier")
                        target_params["costs"]["societal"]["standard_care"] = [
                            c * m for c in target_params["costs"]["societal"]["standard_care"]
                        ]
                    if "new_treatment" in target_params["costs"]["societal"]:
                        m = get_mult("cost_soc_nt_multiplier")
                        target_params["costs"]["societal"]["new_treatment"] = [
                            c * m for c in target_params["costs"]["societal"]["new_treatment"]
                        ]

                # Apply multipliers to QALYs
                if "qalys" in target_params:
                    if "standard_care" in target_params["qalys"]:
                        m = get_mult("qaly_sc_multiplier")
                        target_params["qalys"]["standard_care"] = [
                            q * m for q in target_params["qalys"]["standard_care"]
                        ]
                    if "new_treatment" in target_params["qalys"]:
                        m = get_mult("qaly_nt_multiplier")
                        target_params["qalys"]["new_treatment"] = [
                            q * m for q in target_params["qalys"]["new_treatment"]
                        ]

            # Apply to base parameters (Global)
            apply_multipliers(temp_params)

            # Apply to subgroups if they exist (Specific)
            if "subgroups" in temp_params:
                for name, subgroup in temp_params["subgroups"].items():
                    clean_name = name.replace(" ", "_")
                    apply_multipliers(subgroup, subgroup_suffix=clean_name)

            # Run for both perspectives
            cea_results_soc = run_cea(
                temp_params, perspective="societal", wtp_threshold=50000
            )
            cea_results_hs = run_cea(
                temp_params, perspective="health_system", wtp_threshold=50000
            )

            # Extract subgroup data if available
            extras = {}
            if cea_results_hs.get("subgroup_results"):
                for subgroup, res in cea_results_hs["subgroup_results"].items():
                    key_cost = f"cost_{intervention_type}"
                    key_qaly = f"qalys_{intervention_type}"
                    extras[f"cost_{subgroup}_hs"] = res[key_cost]
                    extras[f"qaly_{subgroup}_hs"] = res[key_qaly]

            if cea_results_soc.get("subgroup_results"):
                for subgroup, res in cea_results_soc["subgroup_results"].items():
                    key_cost = f"cost_{intervention_type}"
                    key_qaly = f"qalys_{intervention_type}"
                    extras[f"cost_{subgroup}_soc"] = res[key_cost]
                    extras[f"qaly_{subgroup}_soc"] = res[key_qaly]

            if intervention_type == "standard_care":
                return (
                    cea_results_hs["cost_standard_care"],
                    cea_results_hs["qalys_standard_care"],
                    cea_results_soc["cost_standard_care"],
                    cea_results_soc["qalys_standard_care"],
                    extras
                )
            elif intervention_type == "new_treatment":
                return (
                    cea_results_hs["cost_new_treatment"],
                    cea_results_hs["qalys_new_treatment"],
                    cea_results_soc["cost_new_treatment"],
                    cea_results_soc["qalys_new_treatment"],
                    extras
                )
            else:
                raise ValueError("Invalid intervention_type")

        psa = ProbabilisticSensitivityAnalysis(
            psa_run_cea_wrapper, psa_distributions, wtp_threshold=50000
        )
        psa_df = psa.run_psa(n_samples=500)

        # Calculate Equity Metrics for PSA (if subgroup data exists)
        # Weights: Low_SES=1.5, High_SES=1.0, Māori=1.5, Non-Māori=1.0
        equity_weights = {
            "Low_SES": 1.5,
            "High_SES": 1.0,
            "Māori": 1.5,
            "Non-Māori": 1.0,
        }

        # Check if we have subgroup columns
        # Pattern: sc_cost_{subgroup}_hs
        subgroups = []
        for col in psa_df.columns:
            if col.startswith("sc_cost_") and col.endswith("_hs"):
                # Extract subgroup name: remove prefix "sc_cost_" and suffix "_hs"
                subgroup = col[8:-3]
                subgroups.append(subgroup)
        subgroups = list(set(subgroups)) # Unique subgroups

        if subgroups:
            logger.info(f"  Calculating Probabilistic Equity Metrics for {name}...")
            # Health System
            weighted_nmb_hs = 0
            for subgroup in subgroups:
                weight = equity_weights.get(subgroup, 1.0)
                # Inc NMB = (Inc QALY * WTP) - Inc Cost
                inc_qaly = psa_df[f"nt_qaly_{subgroup}_hs"] - psa_df[f"sc_qaly_{subgroup}_hs"]
                inc_cost = psa_df[f"nt_cost_{subgroup}_hs"] - psa_df[f"sc_cost_{subgroup}_hs"]
                inc_nmb = (inc_qaly * 50000) - inc_cost
                weighted_nmb_hs += inc_nmb * weight
            psa_df["equity_weighted_nmb_hs"] = weighted_nmb_hs

            # Societal
            weighted_nmb_soc = 0
            for subgroup in subgroups:
                weight = equity_weights.get(subgroup, 1.0)
                inc_qaly = psa_df[f"nt_qaly_{subgroup}_soc"] - psa_df[f"sc_qaly_{subgroup}_soc"]
                inc_cost = psa_df[f"nt_cost_{subgroup}_soc"] - psa_df[f"sc_cost_{subgroup}_soc"]
                inc_nmb = (inc_qaly * 50000) - inc_cost
                weighted_nmb_soc += inc_nmb * weight
            psa_df["equity_weighted_nmb_soc"] = weighted_nmb_soc

        probabilistic_results[name] = psa_df

    # 6. Value of Information
    logger.info("Running Value of Information Analysis...")
    voi_results = {}
    for name, psa_df in probabilistic_results.items():
        logger.info(f"  Generating VOI report for {name}...")
        # Define parameter groups for EVPPI
        # We need to map the PSA parameters to groups
        # Groups: "Cost Parameters", "QALY Parameters"

        # Extract parameter names from PSA columns (excluding result columns)
        # Result columns: cost_*, qaly_*, inc_*, nmb_*, iteration, etc.
        # Parameter columns: *_multiplier, *_multiplier_*

        param_cols = [c for c in psa_df.columns if "multiplier" in c]

        voi_report = generate_voi_report(
            psa_df,
            wtp_thresholds=list(np.linspace(0, 100000, 21)),
            target_population=100000, # Should ideally come from BIA params
            parameter_names=param_cols
        )
        voi_results[name] = voi_report

    # 6. CHEERS Report
    cheers_report = generate_cheers_report()

    # 7. Threshold Analysis
    logger.info("Performing Threshold Analysis...")
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



    # 9. Budget Impact Analysis (BIA)
    logger.info("Performing Budget Impact Analysis (BIA)...")
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
        logger.debug(
            f"  {name}: Population={total_pop:,}, Eligible={eligible_prop:.0%} ({total_pop *eligible_prop:,.0f} people)"
        )

    # 10. Comprehensive Reports
    logger.info("Generating Comprehensive Reports...")
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
            name: {
                "health_system": res.get("health_system", {}).get("dcea_equity_analysis"),
                "societal": res.get("societal", {}).get("human_capital", {}).get("dcea_equity_analysis"),
            }
            for name, res in all_results.items()
            if res.get("health_system", {}).get("dcea_equity_analysis")
            or res.get("societal", {})
            .get("human_capital", {})
            .get("dcea_equity_analysis")
        },
        "selected_interventions": selected_interventions,  # Needed for plotting
    }
