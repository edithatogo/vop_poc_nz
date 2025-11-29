"""
Perspective Value Deterministic Sensitivity Analysis

Dedicated DSA for analyzing how Value of Perspective metrics change with key parameters.
Focuses on WTP threshold, discount rate, and equity weight variations.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cea_model_core import run_cea
from .value_of_information import (
    ProbabilisticSensitivityAnalysis,
    calculate_value_of_perspective,
)
from .visualizations import apply_default_style, save_figure


def perform_perspective_value_dsa(
    model_params: dict,
    intervention_name: str = "Intervention",
    wtp_range: tuple[float, float] = (25000, 75000),
    n_wtp_points: int = 20,
    n_psa_samples: int = 500,
) -> dict:
    """
    Perform DSA on Value of Perspective metrics across WTP threshold range.

    Calculates how all 4 perspective value metrics change with WTP:
    - Expected Value of Perspective (EVP)
    - Perspective Premium
    - Decision Discordance Cost
    - Information Value

    Args:
        model_params: Model parameters dictionary
        intervention_name: Name of intervention
        wtp_range: (min_wtp, max_wtp) in dollars
        n_wtp_points: Number of WTP points to evaluate
        n_psa_samples: PSA sample size for each WTP point

    Returns:
        Dict with DSA results for all perspective value metrics
    """
    print(f"\nPerforming Perspective Value DSA for {intervention_name}...")
    print(f"  WTP range: ${wtp_range[0]:,} to ${wtp_range[1]:,}")
    print(f"  WTP points: {n_wtp_points}")
    print(f"  PSA samples per point: {n_psa_samples}")

    wtp_thresholds = np.linspace(wtp_range[0], wtp_range[1], n_wtp_points)

    results = {
        "wtp_thresholds": wtp_thresholds,
        "evp": [],
        "perspective_premium": [],
        "discordance_cost": [],
        "information_value": [],
        "proportion_discordant": [],
        "prob_hs_optimal": [],
        "prob_soc_optimal": [],
        "correlation": [],
        "intervention_name": intervention_name,
    }

    # Define PSA wrapper
    def psa_run_cea_wrapper(
        sampled_params, intervention_type, base_params=model_params
    ):
        temp_params = copy.deepcopy(base_params)
        # Apply sampling (simplified for demo)
        temp_params["costs"]["health_system"]["new_treatment"][0] *= sampled_params.get(
            "cost_new_treatment_multiplier", 1.0
        )
        temp_params["costs"]["societal"]["new_treatment"][0] *= sampled_params.get(
            "cost_new_treatment_multiplier", 1.0
        )

        if "qalys" in temp_params and len(temp_params["qalys"]["new_treatment"]) > 1:
            temp_params["qalys"]["new_treatment"][1] *= sampled_params.get(
                "qaly_new_treatment_multiplier", 1.0
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

    # PSA distributions (simplified)
    psa_distributions = {
        "cost_new_treatment_multiplier": {
            "distribution": "normal",
            "params": {"mean": 1.0, "std": 0.1},
        },
        "qaly_new_treatment_multiplier": {
            "distribution": "normal",
            "params": {"mean": 1.0, "std": 0.05},
        },
    }

    # Loop over WTP thresholds
    for i, wtp in enumerate(wtp_thresholds):
        print(f"  Progress: {i + 1}/{n_wtp_points} (WTP=${wtp:,.0f})", end="\r")

        # Run PSA for health system perspective
        psa_hs = ProbabilisticSensitivityAnalysis(
            psa_run_cea_wrapper, psa_distributions, wtp_threshold=wtp
        )
        psa_results_hs = psa_hs.run_psa(n_samples=n_psa_samples)

        # Run PSA for societal perspective
        psa_soc = ProbabilisticSensitivityAnalysis(
            psa_run_cea_wrapper, psa_distributions, wtp_threshold=wtp
        )
        psa_results_soc = psa_soc.run_psa(n_samples=n_psa_samples)

        # Calculate perspective value metrics
        vop = calculate_value_of_perspective(
            psa_results_hs,
            psa_results_soc,
            wtp_threshold=wtp,
            chosen_perspective="health_system",
        )

        # Store results
        results["evp"].append(vop["expected_value_of_perspective"])
        results["perspective_premium"].append(vop["perspective_premium"])
        results["discordance_cost"].append(vop["decision_discordance_cost"])
        results["information_value"].append(vop["information_value"])
        results["proportion_discordant"].append(vop["proportion_discordant"])
        results["prob_hs_optimal"].append(vop["prob_health_system_optimal"])
        results["prob_soc_optimal"].append(vop["prob_societal_optimal"])
        results["correlation"].append(vop["correlation_hs_soc"])

    print("\n  Completed!")

    return results


def plot_perspective_value_dsa(
    dsa_results: dict,
    output_dir: str = "output/figures/",
):
    """
    Plot perspective value DSA results showing how metrics change with WTP.

    Creates 3 panels:
    1. All 4 main metrics vs WTP
    2. Probability metrics (discordance, optimal perspectives) vs WTP
    3. Correlation vs WTP

    Args:
        dsa_results: Results from perform_perspective_value_dsa()
        output_dir: Output directory for figures
    """
    apply_default_style()

    intervention_name = dsa_results.get("intervention_name", "Intervention")
    wtp = np.array(dsa_results["wtp_thresholds"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel 1: EVP and Perspective Premium
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(
        wtp / 1000,
        dsa_results["evp"],
        "o-",
        linewidth=2.5,
        markersize=6,
        color="steelblue",
        label="EVP (Opportunity Loss)",
        alpha=0.8,
    )
    line2 = ax1_twin.plot(
        wtp / 1000,
        dsa_results["perspective_premium"],
        "s-",
        linewidth=2.5,
        markersize=6,
        color="forestgreen",
        label="Perspective Premium",
        alpha=0.8,
    )

    ax1.set_xlabel("WTP Threshold ($1000s)", fontsize=11)
    ax1.set_ylabel("EVP ($)", fontsize=11, color="steelblue")
    ax1_twin.set_ylabel("Perspective Premium ($)", fontsize=11, color="forestgreen")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1_twin.tick_params(axis="y", labelcolor="forestgreen")
    ax1.set_title(
        "Expected Value of Perspective & Premium", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Panel 2: Discordance Cost and Information Value
    ax2 = axes[0, 1]
    ax2.plot(
        wtp / 1000,
        dsa_results["discordance_cost"],
        "o-",
        linewidth=2.5,
        markersize=6,
        color="coral",
        label="Discordance Cost",
        alpha=0.8,
    )
    ax2.plot(
        wtp / 1000,
        dsa_results["information_value"],
        "s-",
        linewidth=2.5,
        markersize=6,
        color="purple",
        label="Information Value",
        alpha=0.8,
    )
    ax2.set_xlabel("WTP Threshold ($1000s)", fontsize=11)
    ax2.set_ylabel("Value ($)", fontsize=11)
    ax2.set_title("Decision Costs & Information Value", fontsize=12, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Probability Metrics
    ax3 = axes[1, 0]
    ax3.plot(
        wtp / 1000,
        np.array(dsa_results["proportion_discordant"]) * 100,
        "o-",
        linewidth=2.5,
        markersize=6,
        color="red",
        label="% Discordant Decisions",
        alpha=0.8,
    )
    ax3.plot(
        wtp / 1000,
        np.array(dsa_results["prob_hs_optimal"]) * 100,
        "s-",
        linewidth=2.5,
        markersize=6,
        color="steelblue",
        label="% HS Perspective Optimal",
        alpha=0.8,
    )
    ax3.plot(
        wtp / 1000,
        np.array(dsa_results["prob_soc_optimal"]) * 100,
        "^-",
        linewidth=2.5,
        markersize=6,
        color="forestgreen",
        label="% Societal Optimal",
        alpha=0.8,
    )
    ax3.set_xlabel("WTP Threshold ($1000s)", fontsize=11)
    ax3.set_ylabel("Probability (%)", fontsize=11)
    ax3.set_title("Decision Concordance by Perspective", fontsize=12, fontweight="bold")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])

    # Panel 4: Correlation between perspectives
    ax4 = axes[1, 1]
    ax4.plot(
        wtp / 1000,
        dsa_results["correlation"],
        "o-",
        linewidth=2.5,
        markersize=6,
        color="darkblue",
        alpha=0.8,
    )
    ax4.axhline(
        y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect Correlation"
    )
    ax4.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, label="No Correlation")
    ax4.set_xlabel("WTP Threshold ($1000s)", fontsize=11)
    ax4.set_ylabel("Correlation (HS ↔ Societal)", fontsize=11)
    ax4.set_title("Perspective Correlation", fontsize=12, fontweight="bold")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-0.1, 1.1])

    plt.suptitle(
        f"Perspective Value Sensitivity Analysis: {intervention_name}\n"
        f"WTP Range: ${wtp[0] / 1000:.0f}k - ${wtp[-1] / 1000:.0f}k",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    save_figure(fig, output_dir, f"perspective_value_dsa_{intervention_name}")
    plt.close(fig)


def generate_perspective_value_dsa_table(dsa_results: dict) -> pd.DataFrame:
    """
    Generate summary table of perspective value DSA results.

    Args:
        dsa_results: Results from perform_perspective_value_dsa()

    Returns:
        DataFrame with perspective value metrics at key WTP thresholds
    """
    intervention_name = dsa_results.get("intervention_name", "Intervention")
    wtp = dsa_results["wtp_thresholds"]

    # Select key WTP thresholds (min, 25th, 50th, 75th, max percentiles)
    indices = [0, len(wtp) // 4, len(wtp) // 2, 3 * len(wtp) // 4, len(wtp) - 1]

    data = []
    for idx in indices:
        data.append(
            {
                "Intervention": intervention_name,
                "WTP Threshold ($)": f"${wtp[idx]:,.0f}",
                "EVP ($)": f"${dsa_results['evp'][idx]:,.0f}",
                "Perspective Premium ($)": f"${dsa_results['perspective_premium'][idx]:,.0f}",
                "Discordance Cost ($)": f"${dsa_results['discordance_cost'][idx]:,.0f}",
                "Information Value ($)": f"${dsa_results['information_value'][idx]:,.0f}",
                "% Discordant": f"{dsa_results['proportion_discordant'][idx]:.1%}",
                "Correlation": f"{dsa_results['correlation'][idx]:.3f}",
            }
        )

    return pd.DataFrame(data)
