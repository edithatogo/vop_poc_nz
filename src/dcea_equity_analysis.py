"""
Distributional Cost-Effectiveness Analysis (DCEA) Module.

This module provides functions to assess the equity implications of health interventions
by analyzing the distribution of costs and health outcomes across different
population subgroups.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_gini(net_health_benefits: List[float]) -> float:
    """Calculates the Gini coefficient for a list of net health benefits."""
    array = np.array(net_health_benefits, dtype=np.float64)
    # Gini coefficient calculation
    if np.amin(array) < 0:
        # Values cannot be negative
        array -= np.amin(array)
    # Values cannot be 0
    array += 0.0000001
    # Sort array
    array = np.sort(array)
    # Index array
    index = np.arange(1, array.shape[0] + 1)
    # Number of observations
    n = array.shape[0]
    # Gini coefficient
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def calculate_atkinson_index(
    net_health_benefits: List[float], epsilon: float = 0.5
) -> float:
    """Calculates the Atkinson index for a list of net health benefits."""
    array = np.array(net_health_benefits, dtype=np.float64)
    if epsilon == 1:
        # Special case for epsilon = 1
        return 1 - (np.exp(np.mean(np.log(array))) / np.mean(array))
    else:
        return 1 - ((1 / len(array)) * np.sum(array ** (1 - epsilon))) ** (
            1 / (1 - epsilon)
        ) / np.mean(array)


def apply_equity_weights(
    net_health_benefits: Dict[str, float], weights: Dict[str, float]
) -> float:
    """
    Calculates the equity-weighted total net health benefit.

    Args:
        net_health_benefits: Dictionary of NHB per subgroup.
        weights: Dictionary of equity weights per subgroup.

    Returns:
        Weighted total net health benefit.
    """
    weighted_total = 0.0
    for subgroup, nhb in net_health_benefits.items():
        weight = weights.get(subgroup, 1.0)
        weighted_total += nhb * weight
    return weighted_total


def run_dcea(
    subgroup_results: Dict,
    epsilon: float = 0.5,
    equity_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Performs a distributional cost-effectiveness analysis.

    Args:
        subgroup_results: A dictionary where keys are subgroup names and values are
                          the CEA results for that subgroup.
        epsilon: Inequality aversion parameter for Atkinson index.
        equity_weights: Optional dictionary of weights for each subgroup.

    Returns:
        A dictionary containing the DCEA results, including the distribution of
        net health benefits and an equity impact summary.
    """

    net_health_benefits = {}
    total_health_gain = 0.0

    for subgroup, results in subgroup_results.items():
        # Using incremental NMB as the measure of net health benefit
        net_health_benefits[subgroup] = results["incremental_nmb"]
        total_health_gain += results["incremental_nmb"]

    nhb_list = list(net_health_benefits.values())

    # Calculate weighted health gain if weights are provided
    weighted_health_gain = total_health_gain
    if equity_weights:
        weighted_health_gain = apply_equity_weights(net_health_benefits, equity_weights)

    # For now, a simple summary. More complex equity metrics could be added.
    equity_impact = {
        "distribution_of_net_health_benefits": net_health_benefits,
        "total_health_gain": total_health_gain,
        "weighted_total_health_gain": weighted_health_gain,
        "equity_weights": equity_weights,
        "variance_of_net_health_benefits": np.var(nhb_list),
        "gini_coefficient": calculate_gini(nhb_list),
        "atkinson_index": calculate_atkinson_index(nhb_list, epsilon=epsilon),
        "atkinson_epsilon": epsilon,
    }

    return equity_impact


def generate_dcea_results_table(
    dcea_results: Dict, intervention_name: str
) -> pd.DataFrame:
    """
    Generates a detailed summary table of DCEA results for manuscript inclusion.

    Args:
        dcea_results: The results from the DCEA analysis.
        intervention_name: The name of the intervention.

    Returns:
        A pandas DataFrame with formatted DCEA results.
    """

    rows = []

    # Overall equity metrics
    rows.append(
        {
            "Intervention": intervention_name,
            "Metric": "Total Net Health Benefit",
            "Value": dcea_results["total_health_gain"],
            "Unit": "$",
            "Description": "Sum of incremental net monetary benefits across all subgroups",
        }
    )
    if dcea_results.get("equity_weights"):
        rows.append(
            {
                "Intervention": intervention_name,
                "Metric": "Equity-Weighted Net Health Benefit",
                "Value": dcea_results["weighted_total_health_gain"],
                "Unit": "$",
                "Description": f"Total NMB weighted by equity weights: {dcea_results['equity_weights']}",
            }
        )
    rows.append(
        {
            "Intervention": intervention_name,
            "Metric": "Gini Coefficient",
            "Value": dcea_results["gini_coefficient"],
            "Unit": "",
            "Description": "Measure of inequality in net health benefits distribution (0=perfect equality, 1=perfect inequality)",
        }
    )
    rows.append(
        {
            "Intervention": intervention_name,
            "Metric": f"Atkinson Index (ε={dcea_results.get('atkinson_epsilon', 0.5)})",
            "Value": dcea_results["atkinson_index"],
            "Unit": "",
            "Description": "Measure of inequality, sensitive to transfers at the lower end of the distribution",
        }
    )
    rows.append(
        {
            "Intervention": intervention_name,
            "Metric": "Variance of Net Health Benefits",
            "Value": dcea_results["variance_of_net_health_benefits"],
            "Unit": "$^2$",
            "Description": "Statistical variance of net health benefits across subgroups",
        }
    )

    # Subgroup-specific net health benefits
    for subgroup, nhb_value in dcea_results[
        "distribution_of_net_health_benefits"
    ].items():
        rows.append(
            {
                "Intervention": intervention_name,
                "Metric": f"Net Health Benefit ({subgroup})",
                "Value": nhb_value,
                "Unit": "$",
                "Description": f"Incremental Net Monetary Benefit for subgroup: {subgroup}",
            }
        )

    df = pd.DataFrame(rows)
    return df


def calculate_inequality_aversion_sensitivity(
    subgroup_results: Dict, epsilon_range: List[float] = None
) -> pd.DataFrame:
    """
    Calculates DCEA metrics for a range of inequality aversion parameters (epsilon).

    Args:
        subgroup_results: Dictionary of CEA results per subgroup.
        epsilon_range: List of epsilon values to test. Defaults to 0-10.

    Returns:
        DataFrame containing epsilon, Atkinson Index, and Social Welfare metrics.
    """
    if epsilon_range is None:
        epsilon_range = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    net_health_benefits = {}
    for subgroup, results in subgroup_results.items():
        net_health_benefits[subgroup] = results["incremental_nmb"]

    nhb_list = list(net_health_benefits.values())
    mean_nhb = np.mean(nhb_list)

    results = []
    for eps in epsilon_range:
        atkinson = calculate_atkinson_index(nhb_list, epsilon=eps)
        # Social Welfare Function (SWF) based on Atkinson: W = Mean * (1 - Atkinson)
        # This represents the "Equally Distributed Equivalent" (EDE)
        ede = mean_nhb * (1 - atkinson)

        results.append(
            {
                "epsilon": eps,
                "atkinson_index": atkinson,
                "ede_net_benefit": ede,
                "mean_net_benefit": mean_nhb,
            }
        )

    return pd.DataFrame(results)


def plot_lorenz_curve(
    dcea_results: Dict, intervention_name: str, output_dir: str = "output/figures/"
):
    """
    Generates a Lorenz curve plot.

    This plot shows the distribution of health gains across the population.
    """
    os.makedirs(output_dir, exist_ok=True)

    nhb_list = sorted(dcea_results["distribution_of_net_health_benefits"].values())
    n = len(nhb_list)
    if n == 0:  # pragma: no cover - guard
        total_gain = dcea_results.get("total_health_gain", 0)
        if total_gain == 0:
            print(f"No net health benefits to plot for {intervention_name}.")
            return

    # Ensure positive values for Lorenz curve calculation if some NHB are negative
    nhb_list_positive = [
        max(0, x) for x in nhb_list
    ]  # Take max(0,x) to handle negative NHB

    if (
        np.sum(nhb_list_positive) == 0
    ):  # If all NHB are zero or negative  # pragma: no cover - guard
        print(
            f"Cannot plot Lorenz curve for {intervention_name} as all net health benefits are zero or negative after adjustment."
        )
        return

    lorenz_curve = np.cumsum(nhb_list_positive) / np.sum(nhb_list_positive)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)  # Add (0,0) point

    population_fractions = np.linspace(0, 1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.plot(population_fractions, lorenz_curve, marker="o", label="Lorenz Curve")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Line of Equality")

    ax.set_xlabel("Cumulative Share of Population")
    ax.set_ylabel("Cumulative Share of Net Health Benefit")
    ax.set_title(f"Lorenz Curve for {intervention_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/lorenz_curve_{intervention_name.lower().replace(' ', '_')}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/lorenz_curve_{intervention_name.lower().replace(' ', '_')}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_equity_impact_plane(
    dcea_results: Dict, intervention_name: str, output_dir: str = "output/figures/"
):
    """
    Generates an equity impact plane plot.

    This plot shows the trade-off between total health gains (efficiency) and
    the distribution of those gains (equity).

    Args:
        dcea_results: The results from the DCEA analysis.
        intervention_name: The name of the intervention.
        output_dir: The directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_gain = dcea_results["total_health_gain"]
    # Using Weighted Net Health Benefit as the equity metric (as requested)
    # If weights are not present, this falls back to total gain (neutral)
    equity_metric = dcea_results.get("weighted_total_health_gain", total_gain)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        f"Equity-Efficiency Trade-off: {intervention_name}\n(Societal Perspective, WTP = $50,000/QALY)",
        fontsize=14,
        fontweight="bold",
    )

    ax.scatter(total_gain, equity_metric, s=200, c="blue", alpha=0.7)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # Add 45-degree line to show where Weighted = Unweighted (Neutrality)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k-", alpha=0.5, zorder=0, label="Equity Neutrality")

    ax.set_xlabel("Efficiency (Total Net Health Benefit)", fontsize=12)
    ax.set_ylabel("Equity (Weighted Net Health Benefit)", fontsize=12)
    ax.set_title("Efficiency vs. Equity Trade-off")
    ax.grid(True, alpha=0.3)

    # Annotate quadrants
    # Annotate quadrants relative to the 45-degree line (Neutrality)
    # Points above line = Pro-Equity (Weighted > Unweighted)
    # Points below line = Anti-Equity (Weighted < Unweighted)

    ax.text(
        0.05,
        0.95,
        "Pro-Equity\n(Weighted > Unweighted)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="green",
    )
    ax.text(
        0.95,
        0.05,
        "Anti-Equity\n(Weighted < Unweighted)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="red",
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/equity_impact_plane_{intervention_name.lower().replace(' ', '_')}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/equity_impact_plane_{intervention_name.lower().replace(' ', '_')}.pdf",
        bbox_inches="tight",
    )
    plt.close()
