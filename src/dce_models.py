"""
Discrete Choice Experiment (DCE) Models Module.

This module implements classes and functions for designing and analyzing
Discrete Choice Experiments, including conditional logit and mixed logit models.
Migrated from the legacy dcea_analysis.py.
"""

import warnings
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

class DCEDataProcessor:
    """
    Processes DCE data from stakeholders for health technology assessment preferences.
    """

    def __init__(self):
        self.choice_data: Optional[pd.DataFrame] = None
        self.design_matrix: Optional[pd.DataFrame] = None
        self.attribute_definitions: Optional[Dict[str, Dict]] = None

    def load_choice_data(
        self,
        data_path: str,
        choice_col: str = "choice",
        id_col: str = "respondent_id",
        task_col: str = "choice_task",
    ) -> pd.DataFrame:
        """Load discrete choice data from CSV."""
        self.choice_data = pd.read_csv(data_path)

        required_cols = [id_col, task_col, choice_col]
        missing_cols = [
            col for col in required_cols if col not in self.choice_data.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self._validate_choice_data(choice_col, id_col, task_col)
        return self.choice_data

    def define_attributes(self, attribute_dict: Dict[str, Dict]):
        """Define attributes and their levels for the DCE."""
        self.attribute_definitions = attribute_dict

    def _validate_choice_data(self, choice_col: str, id_col: str, task_col: str):
        """Validate the structure and content of choice data."""
        if self.choice_data is None:
            raise ValueError("Choice data must be loaded before validation")
        unique_choices = self.choice_data[choice_col].unique()
        if not set(unique_choices).issubset({0, 1}):
            raise ValueError(f"Choice column {choice_col} must contain only 0s and 1s")

        grouped = self.choice_data.groupby([id_col, task_col])[choice_col].sum()
        if not all(grouped == 1):
            invalid_tasks = grouped[grouped != 1]
            raise ValueError(
                f"Each choice task should have exactly one selected alternative. Found {len(invalid_tasks)} invalid tasks."
            )

    def prepare_for_modelling(self) -> pd.DataFrame:
        """Prepare the choice data for econometric modeling."""
        if self.choice_data is None:
            raise ValueError("Choice data must be loaded before preparing for modeling")
        return self.choice_data.copy()


class DCEAnalyzer:
    """
    Analyzes discrete choice experiment data using appropriate econometric models.
    """

    def __init__(self, choice_processor: DCEDataProcessor):
        self.processor = choice_processor
        self.model_results: Optional[Dict[str, Any]] = None
        self.coefficients: Optional[Dict[str, Any]] = None

    def fit_conditional_logit(
        self,
        choice_col: str,
        alt_id_col: str,
        attributes: List[str],
        weights: Optional[str] = None,
    ) -> Dict:
        """Fit a conditional logit model to the DCE data using statsmodels."""
        import statsmodels.api as sm

        data = self.processor.prepare_for_modelling().copy()

        if choice_col not in data.columns:
            raise ValueError(f"Choice column '{choice_col}' not found in data")
        if alt_id_col not in data.columns:
            raise ValueError(f"Alternative ID column '{alt_id_col}' not found in data")

        missing = [a for a in attributes if a not in data.columns]
        if missing:
            raise ValueError(f"Missing attribute columns for conditional logit: {missing}")

        y = data[choice_col].astype(int).values
        X = data[attributes].astype(float)

        model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
        res = model.fit()

        self.coefficients = res.params.to_dict()

        results: Dict[str, Any] = {
            "model_type": "conditional_logit_minimal",
            "attributes": attributes,
            "estimated_coefficients": self.coefficients,
            "standard_errors": res.bse.to_dict(),
            "p_values": res.pvalues.to_dict(),
            "log_likelihood": float(res.llf),
        }

        self.model_results = results
        return results

    def analyze_stakeholder_preferences(
        self,
        attribute_importance: bool = True,
        willingness_to_pay: bool = True,
        heterogeneity_analysis: bool = False,
    ) -> Dict:
        """Analyze stakeholder preferences."""
        if self.model_results is None:
            raise ValueError("Model must be fitted before analyzing preferences")

        results: Dict[str, Any] = {
            "attribute_importance": {},
            "willingness_to_pay": {},
            "preference_heterogeneity": {},
            "policy_implications": {},
        }

        if attribute_importance and self.model_results.get("estimated_coefficients"):
            coeffs = self.model_results["estimated_coefficients"] # In minimal model, this is dict
            # If using statsmodels result directly, it might be different, but here we stored dict
            
            abs_coeffs = {k: abs(v) for k, v in coeffs.items() if isinstance(v, (int, float))}
            if abs_coeffs:
                total = sum(abs_coeffs.values())
                if total > 0:
                    results["attribute_importance"] = {
                        k: (v / total) * 100 for k, v in abs_coeffs.items()
                    }

        if willingness_to_pay:
            cost_attr = None
            for attr in self.model_results.get("attributes", []):
                if isinstance(attr, str) and ("cost" in attr.lower() or "price" in attr.lower()):
                    cost_attr = attr
                    break

            wtp_calcs = {}
            if cost_attr and self.model_results.get("estimated_coefficients"):
                cost_coeff = self.model_results["estimated_coefficients"].get(cost_attr, 0)
                if cost_coeff != 0:
                    for attr, coeff in self.model_results["estimated_coefficients"].items():
                        if attr != cost_attr:
                            wtp_calcs[attr] = -coeff / cost_coeff

            results["willingness_to_pay"] = {
                "methodology": "ratio_of_coefficients" if cost_attr else "not_applicable",
                "cost_attribute": cost_attr,
                "wtp_calculations": wtp_calcs,
            }

        results["policy_implications"] = {
            "societal_vs_health_system": "Preferences quantified for different value perspectives",
        }

        return results


def design_dce_study(
    attributes: Dict[str, Dict],
    num_choices: int = 12,
    alternatives_per_task: int = 2,
    num_repeats: int = 2,
) -> pd.DataFrame:
    """Design a discrete choice experiment study."""
    from itertools import product

    attr_names = list(attributes.keys())
    attr_levels = [attributes[attr]["levels"] for attr in attr_names]
    all_combinations = list(product(*attr_levels))

    np.random.seed(42)
    design_size = num_choices * alternatives_per_task
    
    if len(all_combinations) < design_size:
        selected_combinations = np.random.choice(len(all_combinations), size=design_size, replace=True)
    else:
        selected_combinations = np.random.choice(len(all_combinations), size=design_size, replace=False)

    design_data = []
    task_id = 1

    for i in range(0, len(selected_combinations), alternatives_per_task):
        for j in range(alternatives_per_task):
            if i + j < len(selected_combinations):
                combo_idx = selected_combinations[i + j]
                combo = all_combinations[combo_idx]
                row = {"choice_task": task_id, "alternative": j + 1}
                for k, attr_name in enumerate(attr_names):
                    row[attr_name] = combo[k]
                design_data.append(row)
        task_id += 1

    return pd.DataFrame(design_data)


def integrate_dce_with_cea(dce_results: Dict, cea_results: Dict) -> Dict:
    """Integrate DCE results with cost-effectiveness analysis results."""
    integrated_results = {
        "cea_results": cea_results,
        "dce_results": dce_results,
        "integrated_analysis": {
            "cea_favorable": cea_results.get("is_cost_effective", False),
            "dce_alignment_score": np.random.uniform(0.5, 1.0), # Placeholder
        },
        "policy_recommendations": {
            "funding_recommendation": "Consider" if cea_results.get("is_cost_effective", False) else "Defer",
        }
    }
    return integrated_results


def calculate_analytical_capacity_costs(
    dce_study_size: int, stakeholder_groups: List[str], country: str = "NZ"
) -> Dict:
    """Calculate analytical capacity costs for implementing DCE methodology."""
    costs = {
        "study_design": {
            "survey_development": 15000 if country == "NZ" else 12000,
            "experimental_design": 10000 if country == "NZ" else 8000,
            "pilot_testing": 8000 if country == "NZ" else 6000,
        },
        "data_collection": {
            "total_data_collection": dce_study_size * (150 if country == "NZ" else 120),
        },
        "analysis": {
            "statistical_analysis": 25000 if country == "NZ" else 20000,
            "reporting": 10000 if country == "NZ" else 8000,
        },
        "implementation": {
            "process_adaptation": 200000 if country == "NZ" else 180000,
            "staff_training": 30000 if country == "NZ" else 25000,
        },
    }

    total_cost = (
        sum(costs["study_design"].values())
        + costs["data_collection"]["total_data_collection"]
        + sum(costs["analysis"].values())
        + sum(costs["implementation"].values())
    )

    return {
        "cost_breakdown": costs,
        "total_cost": total_cost,
        "funding_entity": "HTA agency (e.g., PHARMAC)",
        "country": country,
    }
