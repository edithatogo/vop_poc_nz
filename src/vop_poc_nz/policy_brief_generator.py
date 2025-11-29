"""
Policy Brief Generator

This module generates a 1-page Executive Summary (Markdown) synthesizing:
1. Efficiency (Health System vs Societal)
2. Equity (Winners/Losers)
3. Fiscal Cost (5-year Budget Impact)
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def generate_policy_brief(
    intervention_results: dict,
    output_dir: str = "output/reports/",
    filename: str = "policy_brief.md",
):
    """
    Generates a policy brief markdown file.
    """
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("# Executive Summary: Health Technology Assessment Policy Brief")
    lines.append(
        "\n**Objective:** To evaluate the value, equity, and fiscal impact of proposed health interventions from a societal perspective."
    )

    lines.append("\n## 1. The Policy Matrix")
    lines.append("Comparison of recommendations across perspectives and impacts.")

    # Build the Policy Matrix Table
    matrix_rows = []
    for name, res in intervention_results.items():
        # Efficiency
        hs_icer = res["health_system"]["icer"]
        soc_icer = res["societal"]["human_capital"][
            "icer"
        ]  # Default to human capital for summary

        hs_decision = "FUND" if res["health_system"]["is_cost_effective"] else "REJECT"
        soc_decision = (
            "FUND"
            if res["societal"]["human_capital"]["is_cost_effective"]
            else "REJECT"
        )

        # Equity
        equity_impact = "Neutral"
        if res.get("dcea_equity_analysis"):
            dcea = res["dcea_equity_analysis"]
            weighted_gain = dcea.get("weighted_total_health_gain", 0)
            unweighted_gain = dcea.get("total_health_gain", 0)
            if weighted_gain > unweighted_gain * 1.05:  # 5% threshold
                equity_impact = "Positive (Pro-Poor)"
            elif weighted_gain < unweighted_gain * 0.95:
                equity_impact = "Negative"

        # Fiscal Impact (Placeholder - would come from BIA if integrated in results)
        # For now, we'll mark as "See BIA" or estimate from costs
        fiscal_impact = "See BIA"

        matrix_rows.append(
            {
                "Intervention": name,
                "Health System Decision": f"{hs_decision} (${hs_icer:,.0f}/QALY)",
                "Societal Decision": f"{soc_decision} (${soc_icer:,.0f}/QALY)",
                "Equity Impact": equity_impact,
                "Fiscal Impact": fiscal_impact,
            }
        )

    df_matrix = pd.DataFrame(matrix_rows)
    lines.append("\n" + df_matrix.to_markdown(index=False))

    lines.append("\n## 2. The Value of Perspective")
    lines.append(
        "Adopting a societal perspective reveals significant opportunity costs (benefits lost) when decisions are restricted to health system costs only."
    )
    lines.append("\n![Discordance Loss](../figures/discordance_loss.png)")

    lines.append("\n## 3. Equity-Efficiency Trade-offs")
    lines.append(
        "Does the societal perspective sacrifice equity? The chart below shows the relationship between total efficiency gains and equity improvements."
    )
    lines.append("\n![Equity Efficiency Plane](../figures/equity_efficiency_plane.png)")

    lines.append("\n## 4. Budget Impact & Affordability")
    lines.append("Five-year fiscal projection for recommended interventions.")
    lines.append("\n![Budget Impact Dashboard](../figures/dashboard_bia.png)")

    lines.append("\n## Recommendation")
    lines.append("> [!IMPORTANT]")
    lines.append(
        "> **Policy Action:** Review interventions where Societal Decision differs from Health System Decision, especially where Equity Impact is Positive."
    )

    # Write to file
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Policy brief generated at {os.path.join(output_dir, filename)}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_results = {
        "Test Intervention": {
            "health_system": {"icer": 60000, "is_cost_effective": False},
            "societal": {"human_capital": {"icer": 40000, "is_cost_effective": True}},
            "dcea_equity_analysis": {
                "weighted_total_health_gain": 120,
                "total_health_gain": 100,
            },
        }
    }
    generate_policy_brief(dummy_results)
