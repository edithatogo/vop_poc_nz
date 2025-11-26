"""
Threshold Analysis Module.

This module provides functions to perform threshold analysis to identify
decision-switching points for cost-effectiveness analysis.
"""

from typing import Dict

import pandas as pd

from .cea_model_core import run_cea


def run_threshold_analysis(
    intervention_name: str,
    base_params: Dict,
    parameter_ranges: Dict,
    wtp_threshold: float = 50000,
) -> Dict:
    """
    Run threshold analysis to identify decision-switching points.

    Parameters:
    - intervention_name: Name of the intervention
    - base_params: Dictionary with base model parameters
    - parameter_ranges: Dictionary with parameter ranges to test
    - wtp_threshold: Willingness-to-pay threshold per QALY

    Returns:
    - Dictionary with threshold analysis results
    """
    threshold_results = {}

    for param_name, param_range in parameter_ranges.items():
        param_results = []

        for param_value in param_range:
            # Modify parameter
            test_params = base_params.copy()

            # Assuming param_name directly refers to a parameter that can be changed
            # For this simplified test, we'll assume a direct modification is sufficient
            # In a real scenario, this would involve a more complex parameter mapping

            # Example: If param_name is 'new_treatment_cost_multiplier'
            if "new_treatment_cost_multiplier" in param_name:
                test_params["costs"]["health_system"]["new_treatment"][0] = (
                    base_params["costs"]["health_system"]["new_treatment"][0]
                    * param_value
                )
                test_params["costs"]["societal"]["new_treatment"][0] = (
                    base_params["costs"]["societal"]["new_treatment"][0] * param_value
                )
            elif "new_treatment_qaly_multiplier" in param_name:
                test_params["qalys"]["new_treatment"][0] = (
                    base_params["qalys"]["new_treatment"][0] * param_value
                )
                test_params["qalys"]["new_treatment"][1] = (
                    base_params["qalys"]["new_treatment"][1] * param_value
                )
            # Add more specific parameter modification logic as needed

            # Run analysis
            hs_result = run_cea(
                test_params, perspective="health_system", wtp_threshold=wtp_threshold
            )
            soc_result = run_cea(
                test_params, perspective="societal", wtp_threshold=wtp_threshold
            )

            hs_cost_effective = hs_result["incremental_nmb"] > 0
            soc_cost_effective = soc_result["incremental_nmb"] > 0

            param_results.append(
                {
                    "parameter_value": param_value,
                    "hs_nmb": hs_result["incremental_nmb"],
                    "soc_nmb": soc_result["incremental_nmb"],
                    "hs_cost_effective": hs_cost_effective,
                    "soc_cost_effective": soc_cost_effective,
                    "discordant": hs_cost_effective != soc_cost_effective,
                }
            )

        threshold_results[param_name] = pd.DataFrame(param_results)

    return threshold_results
