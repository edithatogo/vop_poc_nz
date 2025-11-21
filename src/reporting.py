"""
Automated Reporting Module.

This module provides functions to generate comprehensive reports in Markdown format.
"""

from typing import Dict
import copy
from .cea_model_core import run_cea
from .discordance_analysis import calculate_decision_discordance
from .dcea_equity_analysis import run_dcea

def generate_comprehensive_report(intervention_name: str, params: Dict, wtp_threshold: float = 50000) -> str:
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
    hs_result = run_cea(copy.deepcopy(params), perspective='health_system', wtp_threshold=wtp_threshold)
    soc_result = run_cea(copy.deepcopy(params), perspective='societal', wtp_threshold=wtp_threshold)
    discordance = calculate_decision_discordance(intervention_name, params, wtp_threshold=wtp_threshold)

    # DCEA analysis
    if hs_result and 'subgroup_results' in hs_result and hs_result['subgroup_results']:
        dcea_results = run_dcea(hs_result['subgroup_results'])
        dcea_table = generate_dcea_results_table(dcea_results)
    else:
        dcea_table = "DCEA not applicable for this intervention as no subgroups were defined.\n"

    print(f"DEBUG: Params in generate_comprehensive_report: {params}")
    print(f"DEBUG: Type of params: {type(params)}")
    if isinstance(params, dict):
        print(f"DEBUG: Keys in params: {params.keys()}")

    # Generate report
    report = f"""
# Comprehensive CEA Report: {intervention_name}

## Executive Summary

This report presents a comprehensive cost-effectiveness analysis of {intervention_name} from both health system and societal perspectives.

## Methodology

- **Model Type**: Three-state Markov model
- **Time Horizon**: {params['cycles']} years
- **Discount Rate**: {params['discount_rate'] * 100}%
- **WTP Threshold**: ${wtp_threshold:,.0f} per QALY

## Results

### Health System Perspective
- **ICER**: ${hs_result['icer']:,.2f} per QALY
- **Incremental NMB**: ${hs_result['incremental_nmb']:,.0f}
- **Cost-Effective**: {'Yes' if hs_result['incremental_nmb'] > 0 else 'No'}

### Societal Perspective
- **ICER**: ${soc_result['icer']:,.2f} per QALY
- **Incremental NMB**: ${soc_result['incremental_nmb']:,.0f}
- **Cost-Effective**: {'Yes' if soc_result['incremental_nmb'] > 0 else 'No'}

### Decision Discordance
- **Discordant**: {'Yes' if discordance['discordant'] else 'No'}
- **Loss from Discordance**: ${discordance['loss_from_discordance']:,.0f}
- **Loss in QALYs**: {discordance['loss_qaly']:.2f}

{dcea_table}

## Key Findings

The analysis reveals {'significant differences' if discordance['discordant'] else 'consistency'} between health system and societal perspectives.
{'The societal perspective reveals substantial additional value that is overlooked in the health system perspective.' if discordance['discordant'] else 'Both perspectives yield consistent results.'}

## Recommendations

Based on the societal perspective analysis, {intervention_name} {'should be considered for funding' if (soc_result['incremental_nmb'] > 0) else 'may not be cost-effective at the current WTP threshold'}.
"""

    return report

def generate_dcea_results_table(dcea_results: Dict) -> str:
    """
    Generates a markdown table for DCEA results.

    Args:
        dcea_results: The results from the DCEA analysis.

    Returns:
        A markdown formatted string of the DCEA results.
    """
    if not dcea_results:
        return "DCEA results are not available.\n"

    # Distribution of Net Health Benefits
    nhb_distribution = dcea_results.get('distribution_of_net_health_benefits', {})
    if nhb_distribution:
        nhb_table = "| Subgroup | Net Health Benefit (NMB) |\n| --- | --- |\n"
        for subgroup, nmb in nhb_distribution.items():
            nhb_table += f"| {subgroup} | ${nmb:,.0f} |\n"
    else:
        nhb_table = "No distribution of net health benefits available.\n"

    # Equity Impact Summary
    equity_impact = dcea_results
    total_health_gain = equity_impact.get('total_health_gain', 'N/A')
    if isinstance(total_health_gain, float):
        total_health_gain = f"${total_health_gain:,.0f}"

    variance = equity_impact.get('variance_of_net_health_benefits', 'N/A')
    if isinstance(variance, float):
        variance = f"{variance:,.2f}"

    equity_summary = f"""
### Equity Impact Summary
- **Total Health Gain (Efficiency)**: {total_health_gain}
- **Variance of Net Health Benefits (Equity)**: {variance}
"""

    report = f"""
## Distributional Cost-Effectiveness Analysis (DCEA) Results

### Distribution of Net Health Benefits
{nhb_table}
{equity_summary}
"""
    return report
