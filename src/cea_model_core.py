"""
Corrected and enhanced core CEA model addressing reviewer feedback.

This module implements a robust Markov model for health economic evaluation
with proper mathematical calculations and comprehensive documentation.
"""

import collections.abc
import copy
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import our corrected modules


class MarkovModel:
    """
    A corrected Markov model for health economic evaluation.

    This implementation addresses the mathematical errors identified by
    reviewers and provides transparent, well-documented calculations.
    """

    def __init__(
        self,
        states: List[str],
        transition_matrix: np.ndarray,
        discount_rate: float = 0.03,
    ):
        """
        Initialize the Markov model with proper validation.

        Args:
            states: List of state names (e.g., ['Healthy', 'Sick', 'Dead'])
            transition_matrix: Square matrix of transition probabilities
            discount_rate: Annual discount rate for costs and QALYs (default 3%)
        """
        self.states = states
        self.transition_matrix = np.array(transition_matrix, dtype=float)
        self.discount_rate = discount_rate

        # Validate transition matrix
        self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        """Validate that transition matrix is properly formed."""
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")

        if len(self.states) != self.transition_matrix.shape[0]:
            raise ValueError("Number of states must match matrix dimensions")

        # Check that each row sums to approximately 1
        row_sums = np.sum(self.transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"Transition matrix rows must sum to 1.0, got {row_sums}")

    def run(
        self,
        cycles: int,
        initial_population: np.ndarray,
        costs: np.ndarray,
        qalys: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Runs the Markov model for a given number of cycles.

        This function implements the corrected calculation for costs and QALYs
        with proper discounting, addressing the mathematical errors noted by reviewers.

        Args:
            cycles: Number of model cycles to run
            initial_population: Initial population distribution across states
            costs: Annual costs per person in each state
            qalys: Annual QALYs per person in each state

        Returns:
            Tuple of (total_discounted_cost, total_discounted_qalys)
        """
        num_states = len(self.states)
        initial_population = np.array(initial_population, dtype=float)
        costs = np.array(costs, dtype=float)
        qalys = np.array(qalys, dtype=float)

        if len(initial_population) != num_states:
            raise ValueError(f"Initial population must have {num_states} elements")
        if len(costs) != num_states:
            raise ValueError(f"Costs array must have {num_states} elements")
        if len(qalys) != num_states:
            raise ValueError(f"QALYs array must have {num_states} elements")

        # Initialize population trace
        population_trace = np.zeros((cycles + 1, num_states))
        population_trace[0, :] = initial_population

        total_discounted_cost = 0.0
        total_discounted_qalys = 0.0

        for i in range(cycles):
            # Calculate population for the next cycle
            population_trace[i + 1, :] = population_trace[i, :] @ self.transition_matrix

            # Calculate costs and QALYs for the current cycle
            # Sum across all individuals in all states
            cycle_cost = np.sum(population_trace[i, :] * costs)
            cycle_qalys = np.sum(population_trace[i, :] * qalys)

            # Apply discount rate (discounting at start of period)
            discount_factor = (1 + self.discount_rate) ** i
            total_discounted_cost += cycle_cost / discount_factor
            total_discounted_qalys += cycle_qalys / discount_factor

        return float(total_discounted_cost), float(total_discounted_qalys)


def deep_update(d, u):
    """
    Recursively update a dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def run_cea(
    model_parameters: Dict,
    perspective: str = "health_system",
    wtp_threshold: float = 50000.0,
    productivity_cost_method: str = "human_capital",
) -> Dict:
    """
    Runs a cost-effectiveness analysis, handling subgroups for DCEA if present.

    Args:
        model_parameters: Dictionary containing all model parameters, optionally with a 'subgroups' key.
        perspective: 'health_system' or 'societal'
        wtp_threshold: Willingness-to-pay threshold per QALY
        productivity_cost_method: 'human_capital' or 'friction_cost'

    Returns:
        Dictionary with comprehensive CEA results, including subgroup results if applicable.
    """
    if perspective not in ["health_system", "societal"]:
        raise ValueError("Perspective must be 'health_system' or 'societal'")

    _validate_model_parameters(model_parameters)

    discount_rate = model_parameters.get("discount_rate", 0.03)

    if "subgroups" in model_parameters:
        # Perform DCEA by running CEA for each subgroup
        subgroup_results = {}
        total_cost_sc, total_qalys_sc, total_cost_nt, total_qalys_nt = (
            0.0,
            0.0,
            0.0,
            0.0,
        )

        for subgroup_name, subgroup_params in model_parameters["subgroups"].items():
            # Create a deep copy of the base parameters and update with subgroup-specific values
            subgroup_model_params = copy.deepcopy(model_parameters)
            deep_update(subgroup_model_params, subgroup_params)

            # Ensure discount_rate from parent is preserved
            subgroup_model_params["discount_rate"] = discount_rate

            # Run CEA for the subgroup (recursively, but without the subgroups key to avoid infinite loop)
            if "subgroups" in subgroup_model_params:
                del subgroup_model_params["subgroups"]

            print(  # pragma: no cover - debug tracing
                f"DEBUG: Running subgroup {subgroup_name} with params: {subgroup_model_params.keys()}"
            )
            sub_results = run_cea(
                subgroup_model_params,
                perspective,
                wtp_threshold,
                productivity_cost_method,
            )
            subgroup_results[subgroup_name] = sub_results

            # Aggregate results
            total_cost_sc += sub_results["cost_standard_care"]
            total_qalys_sc += sub_results["qalys_standard_care"]
            total_cost_nt += sub_results["cost_new_treatment"]
            total_qalys_nt += sub_results["qalys_new_treatment"]

        # Use aggregated results for the main CEA calculations
        cost_sc, qalys_sc, cost_nt, qalys_nt = (
            total_cost_sc,
            total_qalys_sc,
            total_cost_nt,
            total_qalys_nt,
        )
        # After aggregation, the subgroup_results dict is part of the final return
        # so we don't nullify it here.

    else:
        # Standard CEA without subgroups
        states = model_parameters["states"]
        tm_standard_care = np.array(
            model_parameters["transition_matrices"]["standard_care"]
        )
        tm_new_treatment = np.array(
            model_parameters["transition_matrices"]["new_treatment"]
        )
        cycles = model_parameters["cycles"]
        initial_population = np.array(model_parameters["initial_population"])

        costs_standard, costs_new, qalys_standard, qalys_new = (
            _get_costs_qalys_by_perspective(
                model_parameters, perspective, productivity_cost_method
            )
        )

        model_sc = MarkovModel(states, tm_standard_care, discount_rate=discount_rate)
        cost_sc, qalys_sc = model_sc.run(
            cycles, initial_population, costs_standard, qalys_standard
        )

        model_nt = MarkovModel(states, tm_new_treatment, discount_rate=discount_rate)
        cost_nt, qalys_nt = model_nt.run(
            cycles, initial_population, costs_new, qalys_new
        )
        subgroup_results = None

    # Calculate incremental values
    inc_cost = float(cost_nt - cost_sc)
    inc_qalys = float(qalys_nt - qalys_sc)
    icer = _calculate_icer(inc_cost, inc_qalys)
    incremental_nmb = (inc_qalys * wtp_threshold) - inc_cost
    is_cost_effective = incremental_nmb > 0

    results = {
        "perspective": perspective,
        "cost_standard_care": cost_sc,
        "qalys_standard_care": qalys_sc,
        "cost_new_treatment": cost_nt,
        "qalys_new_treatment": qalys_nt,
        "incremental_cost": inc_cost,
        "incremental_qalys": inc_qalys,
        "icer": icer,
        "incremental_nmb": incremental_nmb,
        "is_cost_effective": is_cost_effective,
        "wtp_threshold": wtp_threshold,
        "productivity_cost_method": productivity_cost_method,
        "subgroup_results": subgroup_results,  # Include subgroup results if they exist
    }

    return results


def _validate_model_parameters(params: Dict):
    """Validate that model parameters are properly structured."""
    required_keys = [
        "states",
        "transition_matrices",
        "cycles",
        "initial_population",
        "costs",
        "qalys",
    ]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    # Validate structure of transition_matrices
    if (
        "standard_care" not in params["transition_matrices"]
        or "new_treatment" not in params["transition_matrices"]
    ):
        raise ValueError(
            "transition_matrices must contain 'standard_care' and 'new_treatment' keys"
        )

    # Validate structure of costs
    if "health_system" not in params["costs"] or "societal" not in params["costs"]:
        raise ValueError("costs must contain 'health_system' and 'societal' keys")

    for perspective_type in ["health_system", "societal"]:
        if (
            "standard_care" not in params["costs"][perspective_type]
            or "new_treatment" not in params["costs"][perspective_type]
        ):
            raise ValueError(
                f"costs[{perspective_type}] must contain 'standard_care' and 'new_treatment' keys"
            )

    # Validate structure of qalys
    if "standard_care" not in params["qalys"] or "new_treatment" not in params["qalys"]:
        raise ValueError("qalys must contain 'standard_care' and 'new_treatment' keys")


def _calculate_friction_cost(
    model_parameters: Dict, intervention_type: str
) -> np.ndarray:
    """
    Calculate productivity costs using the Friction Cost Method.
    This implementation now considers intervention-specific friction costs if they are defined.
    Source: Based on NZ data for absenteeism and replacement costs.
    """

    # Get base friction cost parameters from the model_parameters or default values
    friction_params = model_parameters.get("friction_cost_params", {})

    # Example default values (these should ideally come from literature/specific studies for NZ)
    default_friction_period_days = 90  # Default friction period in days
    default_replacement_cost_per_day = (
        300  # Default replacement cost per day (e.g., salary/250)
    )
    default_absenteeism_rate = (
        0.04  # Default absenteeism rate (proportion of sick employees)
    )

    # Allow intervention-specific override for friction_cost_params
    intervention_specific_costs = (
        model_parameters.get("costs", {}).get("societal", {}).get(intervention_type, {})
    )
    if isinstance(intervention_specific_costs, dict):
        intervention_specific_friction_params = intervention_specific_costs.get(
            "friction_cost_params", {}
        )
    else:
        intervention_specific_friction_params = {}
    friction_params = {
        **friction_params,
        **intervention_specific_friction_params,
    }  # Merge with overrides

    friction_period_days = friction_params.get(
        "friction_period_days", default_friction_period_days
    )
    replacement_cost_per_day = friction_params.get(
        "replacement_cost_per_day", default_replacement_cost_per_day
    )
    absenteeism_rate = friction_params.get("absenteeism_rate", default_absenteeism_rate)

    # Assuming productivity loss states are defined and associated with absence days
    productivity_loss_states = model_parameters.get(
        "productivity_loss_states", {}
    )  # Now expects a dict {state: absence_days}

    friction_costs_per_state = np.zeros(len(model_parameters["states"]))

    for i, state in enumerate(model_parameters["states"]):
        if state in productivity_loss_states:
            absence_days_per_year = productivity_loss_states[
                state
            ]  # Days of absence for this state

            # Friction cost for this state = replacement_cost_per_day * absenteeism_rate * min(absence_days_per_year, friction_period_days)
            # This calculates the cost incurred due to lost productivity for a given state, constrained by the friction period.
            cost_per_sick_day_in_friction_period = (
                replacement_cost_per_day * absenteeism_rate
            )
            friction_costs_per_state[i] = cost_per_sick_day_in_friction_period * min(
                absence_days_per_year, friction_period_days
            )

    return friction_costs_per_state


def _get_costs_qalys_by_perspective(
    model_parameters: Dict, perspective: str, productivity_cost_method: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract appropriate costs and QALYs based on perspective."""
    if perspective == "health_system":
        costs_standard = np.array(
            model_parameters["costs"]["health_system"]["standard_care"], dtype=float
        )
        costs_new = np.array(
            model_parameters["costs"]["health_system"]["new_treatment"], dtype=float
        )
        qalys_standard = np.array(
            model_parameters["qalys"]["standard_care"], dtype=float
        )
        qalys_new = np.array(model_parameters["qalys"]["new_treatment"], dtype=float)
    elif perspective == "societal":
        # Health system costs
        hs_costs_standard = np.array(
            model_parameters["costs"]["health_system"]["standard_care"], dtype=float
        )
        hs_costs_new = np.array(
            model_parameters["costs"]["health_system"]["new_treatment"], dtype=float
        )

        # Additional societal costs (e.g., out-of-pocket, informal care, and productivity)
        # These are expected to be defined in model_parameters['costs']['societal']
        additional_societal_costs_sc = np.array(
            model_parameters["costs"]["societal"].get(
                "standard_care", np.zeros_like(hs_costs_standard)
            ),
            dtype=float,
        )
        additional_societal_costs_nt = np.array(
            model_parameters["costs"]["societal"].get(
                "new_treatment", np.zeros_like(hs_costs_new)
            ),
            dtype=float,
        )

        # Productivity costs based on the chosen method
        productivity_costs_sc = np.zeros_like(hs_costs_standard)
        productivity_costs_nt = np.zeros_like(hs_costs_new)

        if productivity_cost_method == "human_capital":
            productivity_costs_sc = np.array(
                model_parameters["productivity_costs"]["human_capital"][
                    "standard_care"
                ],
                dtype=float,
            )
            productivity_costs_nt = np.array(
                model_parameters["productivity_costs"]["human_capital"][
                    "new_treatment"
                ],
                dtype=float,
            )
        elif productivity_cost_method == "friction_cost":
            productivity_costs_sc = _calculate_friction_cost(
                model_parameters, "standard_care"
            )
            productivity_costs_nt = _calculate_friction_cost(
                model_parameters, "new_treatment"
            )
        else:
            raise ValueError(
                f"Unknown productivity_cost_method: {productivity_cost_method}. Must be 'human_capital' or 'friction_cost'"
            )

        # Total societal costs: Health System Costs + Additional Societal Costs + Productivity Costs
        costs_standard = (
            hs_costs_standard + additional_societal_costs_sc + productivity_costs_sc
        )
        costs_new = hs_costs_new + additional_societal_costs_nt + productivity_costs_nt

        # For this model, QALYs are the same from both perspectives
        qalys_standard = np.array(
            model_parameters["qalys"]["standard_care"], dtype=float
        )
        qalys_new = np.array(model_parameters["qalys"]["new_treatment"], dtype=float)

    return costs_standard, costs_new, qalys_standard, qalys_new


def _calculate_icer(inc_cost: float, inc_qalys: float) -> Union[float, str]:
    """
    Calculate ICER with proper handling of edge cases.

    Addresses the mathematical errors noted by reviewers.
    """
    if inc_qalys == 0:
        if inc_cost > 0:
            return float("inf")  # New treatment is more costly and no benefit
        elif inc_cost < 0:
            return float("-inf")  # New treatment is cheaper with no additional benefit
        else:
            return 0.0  # No difference in either cost or effectiveness

    icer = inc_cost / inc_qalys

    # Add check for extremely large/small values that might indicate errors
    if abs(icer) > 1e10:  # Very large ICER
        warnings.warn(
            f"Extremely large ICER calculated: {icer}. Verify inputs.", stacklevel=2
        )

    return icer


def _calculate_cer(cost: float, qalys: float) -> Union[float, str]:
    """Calculate cost-effectiveness ratio (cost per QALY) for each intervention."""
    if qalys == 0:
        if cost > 0:
            return float("inf")
        else:
            return 0.0

    return cost / qalys


def create_parameters_table(
    model_parameters: Dict, sources: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create a comprehensive parameters/assumptions/sources table as requested by reviewers.

    Args:
        model_parameters: The model parameters dictionary
        sources: Optional dictionary mapping parameters to their sources

    Returns:
        DataFrame with parameters, values, and sources
    """
    rows = []

    # Add basic parameters
    rows.append(
        {
            "Parameter": "Time horizon (cycles)",
            "Value": model_parameters["cycles"],
            "Description": "Model time horizon",
            "Source": sources.get("cycles", "Model specification")
            if sources
            else "Model specification",
            "Perspective": "All",
        }
    )

    rows.append(
        {
            "Parameter": "Discount rate",
            "Value": model_parameters.get("discount_rate", 0.03),
            "Description": "Annual discount rate for costs and QALYs",
            "Source": sources.get("discount_rate", "Standard practice")
            if sources
            else "Standard practice",
            "Perspective": "All",
        }
    )

    # Add state transition parameters
    for i, state in enumerate(model_parameters["states"]):
        for j, to_state in enumerate(model_parameters["states"]):
            rows.append(
                {
                    "Parameter": f"Transition {state}→{to_state} (Standard Care)",
                    "Value": model_parameters["transition_matrices"]["standard_care"][
                        i
                    ][j],
                    "Description": f"Annual probability of transitioning from {state} to {to_state} under standard care",
                    "Source": sources.get(
                        f"transition_{state}_{to_state}_standard", "Model specification"
                    )
                    if sources
                    else "Model specification",
                    "Perspective": "All",
                }
            )

            rows.append(
                {
                    "Parameter": f"Transition {state}→{to_state} (New Treatment)",
                    "Value": model_parameters["transition_matrices"]["new_treatment"][
                        i
                    ][j],
                    "Description": f"Annual probability of transitioning from {state} to {to_state} with new treatment",
                    "Source": sources.get(
                        f"transition_{state}_{to_state}_new", "Model specification"
                    )
                    if sources
                    else "Model specification",
                    "Perspective": "All",
                }
            )

    # Add cost parameters
    for i, state in enumerate(model_parameters["states"]):
        # Health system costs
        rows.append(
            {
                "Parameter": f"Health system cost - {state} (Standard Care)",
                "Value": model_parameters["costs"]["health_system"]["standard_care"][i],
                "Description": f"Annual health system cost per person in {state} under standard care",
                "Source": sources.get(f"cost_hs_{state}_standard", "Literature")
                if sources
                else "Literature",
                "Perspective": "Health System",
            }
        )

        rows.append(
            {
                "Parameter": f"Health system cost - {state} (New Treatment)",
                "Value": model_parameters["costs"]["health_system"]["new_treatment"][i],
                "Description": f"Annual health system cost per person in {state} with new treatment",
                "Source": sources.get(f"cost_hs_{state}_new", "Literature")
                if sources
                else "Literature",
                "Perspective": "Health System",
            }
        )

        # Societal costs
        rows.append(
            {
                "Parameter": f"Societal cost - {state} (Standard Care)",
                "Value": model_parameters["costs"]["societal"]["standard_care"][i],
                "Description": f"Annual societal cost per person in {state} under standard care (productivity, etc.)",
                "Source": sources.get(f"cost_societal_{state}_standard", "Literature")
                if sources
                else "Literature",
                "Perspective": "Societal",
            }
        )

        rows.append(
            {
                "Parameter": f"Societal cost - {state} (New Treatment)",
                "Value": model_parameters["costs"]["societal"]["new_treatment"][i],
                "Description": f"Annual societal cost per person in {state} with new treatment (productivity, etc.)",
                "Source": sources.get(f"cost_societal_{state}_new", "Literature")
                if sources
                else "Literature",
                "Perspective": "Societal",
            }
        )

    # Add QALY parameters
    for i, state in enumerate(model_parameters["states"]):
        rows.append(
            {
                "Parameter": f"QALY weight - {state} (Standard Care)",
                "Value": model_parameters["qalys"]["standard_care"][i],
                "Description": f"Annual QALY weight per person in {state} under standard care",
                "Source": sources.get(f"qaly_{state}_standard", "Literature")
                if sources
                else "Literature",
                "Perspective": "All",
            }
        )

        rows.append(
            {
                "Parameter": f"QALY weight - {state} (New Treatment)",
                "Value": model_parameters["qalys"]["new_treatment"][i],
                "Description": f"Annual QALY weight per person in {state} with new treatment",
                "Source": sources.get(f"qaly_{state}_new", "Literature")
                if sources
                else "Literature",
                "Perspective": "All",
            }
        )

    df = pd.DataFrame(rows)
    return df


def generate_comparative_icer_table(
    results_health_system: Dict,
    results_societal: Dict,
    intervention_name: str = "Intervention",
) -> pd.DataFrame:
    """
    Generate comparative ICER table as requested by reviewers.

    Args:
        results_health_system: Results from health system perspective analysis
        results_societal: Results from societal perspective analysis
        intervention_name: Name of the intervention being compared

    Returns:
        DataFrame with comparison between perspectives
    """
    data = {
        "Intervention": [intervention_name, intervention_name],
        "Perspective": ["Health System", "Societal"],
        "Cost New Treatment": [
            results_health_system["cost_new_treatment"],
            results_societal["cost_new_treatment"],
        ],
        "Cost Standard Care": [
            results_health_system["cost_standard_care"],
            results_societal["cost_standard_care"],
        ],
        "QALYs New Treatment": [
            results_health_system["qalys_new_treatment"],
            results_societal["qalys_new_treatment"],
        ],
        "QALYs Standard Care": [
            results_health_system["qalys_standard_care"],
            results_societal["qalys_standard_care"],
        ],
        "Incremental Cost": [
            results_health_system["incremental_cost"],
            results_societal["incremental_cost"],
        ],
        "Incremental QALYs": [
            results_health_system["incremental_qalys"],
            results_societal["incremental_qalys"],
        ],
        "ICER ($/QALY)": [results_health_system["icer"], results_societal["icer"]],
        "Net Monetary Benefit": [
            results_health_system["incremental_nmb"],
            results_societal["incremental_nmb"],
        ],
        "Cost-Effective (WTP=$50k)": [
            results_health_system["is_cost_effective"],
            results_societal["is_cost_effective"],
        ],
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":  # pragma: no cover - example usage
    # Example usage with corrected parameters
    print("Running example CEA analysis with corrected calculations...")

    # Example parameters for HPV vaccination (corrected based on manuscript)
    hpv_params = {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.05, 0.05], [0, 0.8, 0.2], [0, 0, 1]],
            "new_treatment": [[0.95, 0.03, 0.02], [0, 0.85, 0.15], [0, 0, 1]],
        },
        "cycles": 20,
        "initial_population": [1000, 0, 0],
        "costs": {
            "health_system": {
                "standard_care": [0, 500, 0],
                "new_treatment": [100, 500, 0],
            },
            "societal": {"standard_care": [0, 2000, 0], "new_treatment": [0, 1000, 0]},
        },
        "qalys": {"standard_care": [1, 0.7, 0], "new_treatment": [1, 0.75, 0]},
    }

    # Run analysis from both perspectives
    hs_results = run_cea(hpv_params, perspective="health_system")
    s_results = run_cea(hpv_params, perspective="societal")

    print("Health System Perspective Results:")
    print(f"  ICER: ${hs_results['icer']:.2f} per QALY")
    print(f"  Incremental NMB: ${hs_results['incremental_nmb']:.2f}")
    print(f"  Cost-effective at $50k/WTP: {hs_results['is_cost_effective']}")

    print("\nSocietal Perspective Results:")
    print(f"  ICER: ${s_results['icer']:.2f} per QALY")
    print(f"  Incremental NMB: ${s_results['incremental_nmb']:.2f}")
    print(f"  Cost-effective at $50k/WTP: {s_results['is_cost_effective']}")

    # Generate comparative ICER table
    comparison_table = generate_comparative_icer_table(
        hs_results, s_results, "HPV Vaccination"
    )
    print("\nComparative ICER Table:")
    print(comparison_table.to_string(index=False))

    # Create parameters table
    sources = {
        "cycles": "Model specification",
        "transition_Healthy_Sick_standard": "Literature",
        "cost_hs_Sick_standard": "New Zealand health system data",
        "cost_societal_Sick_standard": "Productivity loss estimates",
    }
    params_table = create_parameters_table(hpv_params, sources)
    print(f"\nParameters Table (first 10 rows of {len(params_table)}):")
    print(params_table.head(10).to_string(index=False))
