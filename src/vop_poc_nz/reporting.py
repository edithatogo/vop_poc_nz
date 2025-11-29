"""
Automated Reporting Module.

This module provides functions to generate comprehensive reports in Markdown format.
"""

import copy

import pandas as pd

from .cea_model_core import run_cea
from .dcea_equity_analysis import run_dcea
from .discordance_analysis import calculate_decision_discordance

try:  # optional dependency for publication-quality tables
    from great_tables import GT

    GREAT_TABLES_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
    GT = None
    GREAT_TABLES_AVAILABLE = False


def generate_comprehensive_report(
    intervention_name: str, params: dict, wtp_threshold: float = 50000
) -> str:
    """
    Generate a comprehensive report for an intervention.

    Parameters:
    - intervention_name: Name of the intervention
    - params: Dictionary containing model parameters
    - wtp_threshold: Willingness-to-pay threshold per QALY

    Returns:
    - String containing the comprehensive report
    """
    # Run all analyses
    hs_result = run_cea(
        copy.deepcopy(params), perspective="health_system", wtp_threshold=wtp_threshold
    )
    soc_result = run_cea(
        copy.deepcopy(params), perspective="societal", wtp_threshold=wtp_threshold
    )
    discordance = calculate_decision_discordance(
        intervention_name, params, wtp_threshold=wtp_threshold
    )

    # DCEA analysis
    if hs_result and "subgroup_results" in hs_result and hs_result["subgroup_results"]:
        dcea_results = run_dcea(hs_result["subgroup_results"])
        dcea_table = generate_dcea_results_table(dcea_results)
    else:
        dcea_table = (
            "DCEA not applicable for this intervention as no subgroups were defined.\n"
        )

    # Generate report
    report = f"""
# Comprehensive CEA Report: {intervention_name}

## Executive Summary

This report presents a comprehensive cost-effectiveness analysis of {intervention_name} from both health system and societal perspectives.

## Methodology

- **Model Type**: Three-state Markov model
- **Time Horizon**: {params["cycles"]} years
- **Discount Rate**: {params.get("discount_rate", 0.03) * 100}%
- **WTP Threshold**: ${wtp_threshold:,.0f} per QALY

## Results

### Health System Perspective
- **ICER**: ${hs_result["icer"]:,.2f} per QALY
- **Incremental NMB**: ${hs_result["incremental_nmb"]:,.0f}
- **Cost-Effective**: {"Yes" if hs_result["incremental_nmb"] > 0 else "No"}

### Societal Perspective
- **ICER**: ${soc_result["icer"]:,.2f} per QALY
- **Incremental NMB**: ${soc_result["incremental_nmb"]:,.0f}
- **Cost-Effective**: {"Yes" if soc_result["incremental_nmb"] > 0 else "No"}

### Decision Discordance
- **Discordant**: {"Yes" if discordance["discordant"] else "No"}
- **Preferred Perspective**: {discordance["preferred_perspective"].replace("_", " ").title()}
- **Opportunity Loss (NMB)**: ${discordance["loss_from_discordance"]:,.0f}
- **Opportunity Loss (QALYs)**: {discordance["loss_qaly"]:.2f}

## Impact Inventory (CHEERS 2022)

{generate_impact_inventory_table(params)}

{dcea_table}

## Key Findings

The analysis reveals {"significant differences" if discordance["discordant"] else "consistency"} between health system and societal perspectives.
{"The societal perspective reveals substantial additional value that is overlooked in the health system perspective." if discordance["discordant"] else "Both perspectives yield consistent results."}

## Recommendations

Based on the societal perspective analysis, {intervention_name} {"should be considered for funding" if (soc_result["incremental_nmb"] > 0) else "may not be cost-effective at the current WTP threshold"}.
"""

    return report


def generate_dcea_results_table(dcea_results: dict) -> str:
    """
    Generates a markdown table for DCEA results.

    Args:
        dcea_results: The results from the DCEA analysis.

    Returns:
        A markdown formatted string of the DCEA results.
    """
    if not dcea_results:
        return "DCEA results are not available.\n"

    rows = []

    nhb_distribution = dcea_results.get("distribution_of_net_health_benefits", {})
    for subgroup, nmb in nhb_distribution.items():
        rows.append(
            {
                "Subgroup": subgroup,
                "Net Health Benefit (NMB)": nmb,
                "Type": "Distribution",
            }
        )

    equity_impact = dcea_results
    rows.append(
        {
            "Subgroup": "Overall",
            "Net Health Benefit (NMB)": equity_impact.get("total_health_gain", 0.0),
            "Type": "Total Health Gain",
        }
    )
    rows.append(
        {
            "Subgroup": "Overall",
            "Net Health Benefit (NMB)": equity_impact.get(
                "variance_of_net_health_benefits", 0.0
            ),
            "Type": "Variance of Net Health Benefits",
        }
    )
    rows.append(
        {
            "Subgroup": "Overall",
            "Net Health Benefit (NMB)": equity_impact.get("gini_coefficient", 0.0),
            "Type": "Gini Coefficient",
        }
    )

    df = pd.DataFrame(rows)
    return _render_table(df)


def generate_impact_inventory_table(params: dict) -> str:
    """
    Generate a CHEERS-style impact inventory indicating which cost components
    are included under each perspective.
    """
    rows = []
    categories = [
        ("Direct medical (health system)", "health_system"),
        ("Direct non-medical / societal", "societal"),
        ("Productivity (human capital)", "productivity_hc"),
        ("Productivity (friction cost)", "productivity_fc"),
    ]

    costs = params.get("costs", {})
    productivity = params.get("productivity_costs", {})

    hs_costs = costs.get("health_system", {})
    soc_costs = costs.get("societal", {})
    hc_costs = productivity.get("human_capital", {})

    for label, key in categories:
        if key == "health_system":
            hs_included = bool(hs_costs)
            soc_included = False
        elif key == "societal":
            hs_included = False
            soc_included = bool(soc_costs)
        elif key == "productivity_hc":
            hs_included = False
            soc_included = bool(hc_costs)
        else:
            # friction cost method is available if friction parameters are defined
            hs_included = False
            soc_included = bool(params.get("friction_cost_params"))

        rows.append(
            {
                "Category": label,
                "Health System Included": "Yes" if hs_included else "No",
                "Societal Included": "Yes" if soc_included else "No",
            }
        )

    df = pd.DataFrame(rows)
    return _render_table(df)


def _render_table(df: pd.DataFrame) -> str:
    """Render a table using great_tables when available, else markdown."""
    if (
        GREAT_TABLES_AVAILABLE and GT is not None
    ):  # pragma: no cover - optional dependency
        try:
            return GT(df).to_html()
        except Exception:
            # Fall back silently to markdown if rendering fails
            pass
    return df.to_markdown(index=False)
