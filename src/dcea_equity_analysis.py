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

    # Extract Societal results (default for Lorenz)
    if "societal" in dcea_results:
        dcea_results = dcea_results["societal"]
    
    if not dcea_results:
        print(f"No societal DCEA results for {intervention_name}")
        return

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
    dcea_results_dict: Dict, intervention_name: str, output_dir: str = "output/figures/"
):
    """
    Generates a comparative equity impact plane plot (Health System vs Societal).

    Args:
        dcea_results_dict: Dictionary containing 'health_system' and 'societal' DCEA results.
        intervention_name: The name of the intervention.
        output_dir: The directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.suptitle(
        f"Equity-Efficiency Trade-off: {intervention_name}",
        fontsize=16,
        fontweight="bold",
    )

    perspectives = [
        ("Health System", dcea_results_dict.get("health_system"), ax1),
        ("Societal", dcea_results_dict.get("societal"), ax2),
    ]

    for title, results, ax in perspectives:
        if not results:
            ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
            ax.set_title(f"{title} Perspective")
            continue

        total_gain = results["total_health_gain"]
        equity_metric = results.get("weighted_total_health_gain", total_gain)

        ax.scatter(total_gain, equity_metric, s=200, c="blue", alpha=0.7)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        # Add 45-degree line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        # Ensure limits include 0
        lims[0] = min(lims[0], -1000)
        lims[1] = max(lims[1], 1000)
        
        ax.plot(lims, lims, "k-", alpha=0.5, zorder=0, label="Equity Neutrality")

        ax.set_xlabel("Efficiency (Total Net Health Benefit)", fontsize=12)
        ax.set_ylabel("Equity (Weighted Net Health Benefit)", fontsize=12)
        ax.set_title(f"{title} Perspective", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Annotate quadrants
        ax.text(
            0.05, 0.95, "Pro-Equity\n(Weighted > Unweighted)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10, color="green",
        )
        ax.text(
            0.95, 0.05, "Anti-Equity\n(Weighted < Unweighted)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="red",
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Make room for suptitle
    
    filename = f"equity_impact_plane_{intervention_name.lower().replace(' ', '_')}"
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches="tight")
    plt.close()


def plot_comparative_equity_impact_plane(
    all_dcea_results: Dict, output_dir: str = "output/figures/"
):
    """
    Generates a single comparative equity impact plane for ALL interventions.
    
    Two subplots: Health System vs Societal.
    Each intervention is a distinct point/marker.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    fig.suptitle(
        "Comparative Equity-Efficiency Trade-offs",
        fontsize=16,
        fontweight="bold",
    )
    
    # Define styles for interventions
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Helper to plot a perspective
    def plot_perspective(ax, perspective_key, title):
        # Add quadrants and lines first
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        
        # 45-degree line placeholder (will update limits later)
        ax.plot([-1e9, 1e9], [-1e9, 1e9], "k-", alpha=0.3, zorder=0, label="Equity Neutrality")
        
        # Plot points
        texts = []
        max_val = 0
        min_val = 0
        
        for i, (name, results_dict) in enumerate(all_dcea_results.items()):
            res = results_dict.get(perspective_key)
            if not res:
                continue
                
            total_gain = res["total_health_gain"]
            equity_metric = res.get("weighted_total_health_gain", total_gain)
            
            # Update range tracking
            max_val = max(max_val, abs(total_gain), abs(equity_metric))
            
            ax.scatter(
                total_gain, 
                equity_metric, 
                s=150, 
                label=name,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9,
                zorder=10
            )
            
        # Set limits symmetric and large enough
        limit = max_val * 1.2 if max_val > 0 else 1000
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        ax.set_xlabel("Efficiency (Total Net Health Benefit)", fontsize=12)
        ax.set_ylabel("Equity (Weighted Net Health Benefit)", fontsize=12)
        ax.set_title(f"{title} Perspective", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Annotate quadrants
        ax.text(0.05, 0.95, "Pro-Equity\n(Weighted > Unweighted)", transform=ax.transAxes, ha="left", va="top", color="green", fontsize=10)
        ax.text(0.95, 0.05, "Anti-Equity\n(Weighted < Unweighted)", transform=ax.transAxes, ha="right", va="bottom", color="red", fontsize=10)

    plot_perspective(ax1, "health_system", "Health System")
    plot_perspective(ax2, "societal", "Societal")
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    # Filter out the 'Equity Neutrality' line from legend if desired, or keep it
    # Let's keep distinct intervention labels
    by_label = dict(zip(labels, handles))
    if "Equity Neutrality" in by_label:
        del by_label["Equity Neutrality"] # Remove line from legend to focus on interventions
        
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    plt.savefig(f"{output_dir}/equity_impact_plane_comparative.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/equity_impact_plane_comparative.pdf", bbox_inches="tight")
    plt.close()


def plot_probabilistic_equity_impact_plane(
    probabilistic_results: Dict[str, pd.DataFrame], output_dir: str = "output/figures/"
):
    """
    Generates a probabilistic equity impact plane (Scatter Plot of PSA results).
    
    Plots the cloud of (Efficiency, Equity) points for each intervention.
    Efficiency = Total Net Health Benefit (Unweighted)
    Equity = Equity-Weighted Net Health Benefit
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    fig.suptitle(
        "Probabilistic Equity Impact Plane (PSA Scatter)",
        fontsize=16,
        fontweight="bold",
    )
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_perspective(ax, perspective_suffix, title):
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.plot([-1e9, 1e9], [-1e9, 1e9], "k-", alpha=0.3, zorder=0, label="Equity Neutrality")
        
        max_val = 0
        
        for i, (name, df) in enumerate(probabilistic_results.items()):
            eff_col = f"inc_nmb_{perspective_suffix}"
            eq_col = f"equity_weighted_nmb_{perspective_suffix}"
            
            if eff_col not in df.columns or eq_col not in df.columns:
                continue
                
            efficiency = df[eff_col]
            equity = df[eq_col]
            
            # Update range
            max_val = max(max_val, efficiency.abs().max(), equity.abs().max())
            
            # Plot scatter cloud
            ax.scatter(
                efficiency, 
                equity, 
                s=10, 
                alpha=0.1, 
                color=colors[i % len(colors)],
                label=name if i < 10 else None # Avoid legend clutter if too many
            )
            
            # Plot centroid
            ax.scatter(
                efficiency.mean(), 
                equity.mean(), 
                s=100, 
                marker='X',
                edgecolor='white',
                linewidth=1.5,
                color=colors[i % len(colors)],
                zorder=10
            )

        limit = max_val * 1.1 if max_val > 0 else 1000
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        ax.set_xlabel("Efficiency (Incremental Net Health Benefit)", fontsize=12)
        ax.set_ylabel("Equity (Equity-Weighted Net Health Benefit)", fontsize=12)
        ax.set_title(f"{title} Perspective", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Annotate
        ax.text(0.05, 0.95, "Pro-Equity", transform=ax.transAxes, ha="left", va="top", color="green", fontsize=10)
        ax.text(0.95, 0.05, "Anti-Equity", transform=ax.transAxes, ha="right", va="bottom", color="red", fontsize=10)

    plot_perspective(ax1, "hs", "Health System")
    plot_perspective(ax2, "soc", "Societal")
    
    # Legend (using centroids proxy handles)
    handles = []
    labels = []
    for i, name in enumerate(probabilistic_results.keys()):
        h = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)], markersize=10)
        handles.append(h)
        labels.append(name)
        
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    plt.savefig(f"{output_dir}/equity_impact_plane_probabilistic.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/equity_impact_plane_probabilistic.pdf", bbox_inches="tight")
    plt.close()


def plot_probabilistic_equity_impact_plane_with_delta(
    probabilistic_results: Dict[str, pd.DataFrame], output_dir: str = "output/figures/"
):
    """
    Generates a probabilistic equity impact plane with Delta subplot.
    3 Subplots: Health System, Societal, Delta (Societal - HS).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(
        "Probabilistic Equity Impact Plane with Perspective Delta\n(PSA Scatter)",
        fontsize=16,
        fontweight="bold",
    )
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    perspectives = ["Health System", "Societal", "Delta (Societal - HS)"]
    
    # Helper to plot a perspective
    def plot_perspective(ax, perspective_idx, title):
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        
        # Only plot 45-degree line for HS and Soc, not necessarily for Delta (though maybe relevant)
        if perspective_idx < 2:
            ax.plot([-1e9, 1e9], [-1e9, 1e9], "k-", alpha=0.3, zorder=0, label="Equity Neutrality")
        
        max_val = 0
        
        for i, (name, df) in enumerate(probabilistic_results.items()):
            if perspective_idx == 0: # HS
                eff_col = "inc_nmb_hs"
                eq_col = "equity_weighted_nmb_hs"
            elif perspective_idx == 1: # Soc
                eff_col = "inc_nmb_soc"
                eq_col = "equity_weighted_nmb_soc"
            else: # Delta
                # Calculate delta
                eff_hs = df.get("inc_nmb_hs", pd.Series(0, index=df.index))
                eq_hs = df.get("equity_weighted_nmb_hs", pd.Series(0, index=df.index))
                eff_soc = df.get("inc_nmb_soc", pd.Series(0, index=df.index))
                eq_soc = df.get("equity_weighted_nmb_soc", pd.Series(0, index=df.index))
                
                efficiency = eff_soc - eff_hs
                equity = eq_soc - eq_hs
                
                # Skip check below as we constructed series
                # But check if original cols existed
                if "inc_nmb_hs" not in df.columns: continue
                
                # Plot
                max_val = max(max_val, efficiency.abs().max(), equity.abs().max())
                
                ax.scatter(efficiency, equity, s=10, alpha=0.1, color=colors[i % len(colors)])
                ax.scatter(efficiency.mean(), equity.mean(), s=100, marker='X', edgecolor='white', linewidth=1.5, color=colors[i % len(colors)], zorder=10)
                continue

            if eff_col not in df.columns or eq_col not in df.columns:
                continue
                
            efficiency = df[eff_col]
            equity = df[eq_col]
            
            # Update range
            max_val = max(max_val, efficiency.abs().max(), equity.abs().max())
            
            # Plot scatter cloud
            ax.scatter(
                efficiency, 
                equity, 
                s=10, 
                alpha=0.1, 
                color=colors[i % len(colors)],
            )
            
            # Plot centroid
            ax.scatter(
                efficiency.mean(), 
                equity.mean(), 
                s=100, 
                marker='X',
                edgecolor='white',
                linewidth=1.5,
                color=colors[i % len(colors)],
                zorder=10
            )

        limit = max_val * 1.1 if max_val > 0 else 1000
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        if perspective_idx < 2:
            ax.set_xlabel("Efficiency (Incremental Net Health Benefit)", fontsize=12)
            ax.set_ylabel("Equity (Equity-Weighted Net Health Benefit)", fontsize=12)
        else:
            ax.set_xlabel("Delta Efficiency (Soc - HS)", fontsize=12)
            ax.set_ylabel("Delta Equity (Soc - HS)", fontsize=12)
            
        ax.set_title(f"{title}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Annotate
        if perspective_idx < 2:
            ax.text(0.05, 0.95, "Pro-Equity", transform=ax.transAxes, ha="left", va="top", color="green", fontsize=10)
            ax.text(0.95, 0.05, "Anti-Equity", transform=ax.transAxes, ha="right", va="bottom", color="red", fontsize=10)

    plot_perspective(axes[0], 0, "Health System")
    plot_perspective(axes[1], 1, "Societal")
    plot_perspective(axes[2], 2, "Delta (Societal - HS)")
    
    # Legend (using centroids proxy handles)
    handles = []
    labels = []
    for i, name in enumerate(probabilistic_results.keys()):
        h = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)], markersize=10)
        handles.append(h)
        labels.append(name)
        
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(labels), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    plt.savefig(f"{output_dir}/equity_impact_plane_probabilistic_with_delta.png", bbox_inches="tight")
    plt.close()
