"""
Decision Discordance Analysis Module.

This module provides functions to calculate and analyze decision discordance
between different perspectives in cost-effectiveness analysis.
"""

from typing import Dict

from .cea_model_core import run_cea


def calculate_decision_discordance(
    intervention_name: str, params: Dict, wtp_threshold: float = 50000
) -> Dict:
    """
    Calculate decision discordance metrics.

    Parameters:
    - intervention_name: Name of the intervention
    - params: Dictionary containing model parameters
    - wtp_threshold: Willingness-to-pay threshold per QALY

    Returns:
    - Dictionary with discordance metrics
    """
    hs_result = run_cea(
        params, perspective="health_system", wtp_threshold=wtp_threshold
    )
    soc_result = run_cea(params, perspective="societal", wtp_threshold=wtp_threshold)

    hs_cost_effective = hs_result["incremental_nmb"] > 0
    soc_cost_effective = soc_result["incremental_nmb"] > 0

    discordant = hs_cost_effective != soc_cost_effective

    loss_from_discordance = 0.0
    if (discordant and not soc_cost_effective) or (discordant and soc_cost_effective):  # pragma: no cover - guard
        loss_from_discordance = abs(float(soc_result["incremental_nmb"]))

    return {
        "intervention": intervention_name,
        "discordant": discordant,
        "hs_cost_effective": hs_cost_effective,
        "soc_cost_effective": soc_cost_effective,
        "loss_from_discordance": loss_from_discordance,
        "loss_qaly": loss_from_discordance / wtp_threshold,
    }
