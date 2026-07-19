"""
Decision Discordance Analysis Module.

This module provides functions to calculate and analyze decision discordance
between different perspectives in cost-effectiveness analysis.
"""

import numpy as np

from .cea_model_core import run_cea
from .perspective import DecisionRule, NetBenefitTensor, TiePolicy


def calculate_decision_discordance(
    intervention_name: str, params: dict, wtp_threshold: float = 50000
) -> dict:
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

    # A perspective is an evaluative lens, not an alternative. Represent the
    # status quo and intervention as strategies, select one fixed strategy under
    # each lens, then evaluate the health-system choice under the societal lens.
    tensor = NetBenefitTensor(
        np.array(
            [
                [
                    [0.0, 0.0],
                    [
                        float(hs_result["incremental_nmb"]),
                        float(soc_result["incremental_nmb"]),
                    ],
                ]
            ]
        ),
        strategies=("standard_care", "intervention"),
        perspectives=("health_system", "societal"),
        case_id=intervention_name,
        attrs={"wtp_threshold": wtp_threshold},
    )
    regret = tensor.evop(
        choose_under="health_system",
        evaluate_under="societal",
        decision_rule=DecisionRule.EXPECTED_VALUE,
        selection_tie_policy=TiePolicy.FIRST,
    )
    loss_from_discordance = regret.per_person

    return {
        "intervention": intervention_name,
        "discordant": discordant,
        "hs_cost_effective": hs_cost_effective,
        "soc_cost_effective": soc_cost_effective,
        "loss_from_discordance": loss_from_discordance,
        "loss_qaly": loss_from_discordance / wtp_threshold if wtp_threshold else 0.0,
        "preferred_perspective": "societal" if discordant else "health_system",
        "chosen_strategy": regret.chosen_strategy,
        "target_strategy": regret.target_strategy,
        "choose_under": regret.choose_under,
        "evaluate_under": regret.evaluate_under,
        "decision_rule": regret.decision_rule.value,
        "tie_policy": regret.selection_tie_policy.value,
        "method_contract_version": regret.method_contract_version,
    }
