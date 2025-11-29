"""
Comparative Multi-Intervention Visualization Functions

Functions for comparing multiple interventions side-by-side on the same plot:
1. Comparative cash flow (lines)
2. ICER ladder comparison
3. Net Monetary Benefit comparison
4. Equity impact comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_comparative_cash_flow(
    bia_results: dict[str, pd.DataFrame],
    output_dir: str = "output/figures/",
    discount: bool = True,
):
    """
    Plot comparative cash flow for all interventions on one chart (line plot).

    Args:
        bia_results: Dict mapping intervention names to BIA DataFrames
        output_dir: Output directory
        discount: Whether to use discounted (True) or nominal (False) costs
    """
    from .visualizations import apply_default_style, save_figure

    apply_default_style()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=1200)

    colors = plt.cm.tab10(np.linspace(0, 1, len(bia_results)))

    for idx, (intervention_name, bia_df) in enumerate(bia_results.items()):
        years = bia_df["year"].values

        if discount and "discounted_net_cost" in bia_df.columns:
            net_costs = bia_df["discounted_net_cost"].values
            ylabel = "Net Budget Impact (Discounted, $)"
        else:
            net_costs = bia_df["net_cost"].values
            ylabel = "Net Budget Impact (Nominal, $)"

        ax.plot(
            years,
            net_costs,
            "o-",
            linewidth=2.5,
            markersize=8,
            label=intervention_name,
            color=colors[idx],
            alpha=0.8,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        "Comparative Budget Impact Analysis\nAll Interventions",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f"${x / 1e6:.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:.0f}K"
        )
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "comparative_cash_flow")
    plt.close(fig)


def plot_icer_ladder(
    intervention_results: dict[str, dict],
    wtp_threshold: float = 50000,
    output_dir: str = "output/figures/",
    perspective: str = "societal",
):
    """
    Plot ICER ladder comparing all interventions on incremental cost-effectiveness plane.

    Shows the "frontier" of non-dominated interventions.

    Args:
        intervention_results: Dict of intervention names to CEA results
        wtp_threshold: WTP threshold (for reference line)
        output_dir: Output directory
        perspective: Which perspective to use
    """
    from .visualizations import apply_default_style, save_figure

    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1200)

    # Extract data
    interventions = []
    inc_costs = []
    inc_qalys = []
    icers = []

    for name, results in intervention_results.items():
        if perspective in results:
            if (
                isinstance(results[perspective], dict)
                and "incremental_cost" in results[perspective]
            ):
                data = results[perspective]
            elif "incremental_cost" in results:
                data = results
            else:
                continue

            interventions.append(name)
            inc_costs.append(data.get("incremental_cost", 0))
            inc_qalys.append(data.get("incremental_qalys", 0))
            icers.append(data.get("icer", np.inf))

    # Sort by incremental QALYs
    sorted_data = sorted(zip(inc_qalys, inc_costs, interventions, icers))
    inc_qalys_sorted, inc_costs_sorted, interventions_sorted, icers_sorted = zip(
        *sorted_data
    )

    # Plot points
    colors = plt.cm.viridis(np.linspace(0, 1, len(interventions)))
    for i, (qaly, cost, name, _icer) in enumerate(
        zip(inc_qalys_sorted, inc_costs_sorted, interventions_sorted, icers_sorted)
    ):
        ax.scatter(
            qaly,
            cost,
            s=300,
            color=colors[i],
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            name,
            (qaly, cost),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": colors[i], "alpha": 0.3},
        )

    # Connect efficient frontier
    ax.plot(
        inc_qalys_sorted,
        inc_costs_sorted,
        "k--",
        linewidth=1.5,
        alpha=0.5,
        zorder=2,
        label="Frontier",
    )

    # WTP threshold line
    max_qaly = max(inc_qalys_sorted) if inc_qalys_sorted else 1
    ax.plot(
        [0, max_qaly],
        [0, max_qaly * wtp_threshold],
        "r--",
        linewidth=2,
        alpha=0.6,
        label=f"WTP = ${wtp_threshold:,}/QALY",
    )

    # Quadrant labels
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Incremental QALYs", fontsize=12)
    ax.set_ylabel("Incremental Cost ($)", fontsize=12)
    ax.set_title(
        f"ICER Ladder: {perspective.replace('_', ' ').title()} Perspective",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, f"icer_ladder_{perspective}")
    plt.close(fig)


def plot_nmb_comparison(
    intervention_results: dict[str, dict],
    wtp_threshold: float = 50000,
    output_dir: str = "output/figures/",
    perspectives: list[str] = None,
):
    """
    Bar chart comparing Net Monetary Benefit across interventions and perspectives.

    Args:
        intervention_results: Dict of intervention names to results
        wtp_threshold: WTP threshold for NMB calculation
        output_dir: Output directory
        perspectives: List of perspectives to compare (default: both)
    """
    from .visualizations import apply_default_style, save_figure

    apply_default_style()

    if perspectives is None:
        perspectives = ["health_system", "societal"]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=1200)

    interventions = list(intervention_results.keys())
    x = np.arange(len(interventions))
    width = 0.35

    nmb_data = {persp: [] for persp in perspectives}

    for intervention_name in interventions:
        results = intervention_results[intervention_name]
        for persp in perspectives:
            if persp in results:
                if (
                    isinstance(results[persp], dict)
                    and "incremental_nmb" in results[persp]
                ):
                    nmb = results[persp]["incremental_nmb"]
                elif "incremental_nmb" in results:
                    nmb = results["incremental_nmb"]
                else:
                    nmb = 0
            else:
                nmb = 0
            nmb_data[persp].append(nmb)

    # Plot bars
    colors = {"health_system": "steelblue", "societal": "forestgreen"}
    for i, persp in enumerate(perspectives):
        offset = width * (i - len(perspectives) / 2 + 0.5)
        bars = ax.bar(
            x + offset,
            nmb_data[persp],
            width,
            label=persp.replace("_", " ").title(),
            color=colors.get(persp, f"C{i}"),
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels on bars
        for _idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${height / 1e3:.0f}K"
                if abs(height) < 1e6
                else f"${height / 1e6:.1f}M",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Intervention", fontsize=12)
    ax.set_ylabel(f"Net Monetary Benefit @ WTP=${wtp_threshold:,}/QALY", fontsize=12)
    ax.set_title(
        "Comparative Net Monetary Benefit\nAcross Interventions and Perspectives",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(interventions, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "nmb_comparison")
    plt.close(fig)


def plot_equity_impact_comparison(
    intervention_results: dict[str, dict], output_dir: str = "output/figures/"
):
    """
    Compare equity impact (Atkinson index) across interventions.

    Args:
        intervention_results: Dict of intervention names to results
        output_dir: Output directory
    """
    from .visualizations import apply_default_style, save_figure

    apply_default_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=1200)

    interventions = []
    atkinson_indices = []
    gini_coefficients = []
    weighted_nmb = []

    for name, results in intervention_results.items():
        # Navigate to DCEA results
        if "societal" in results:
            societal = results["societal"]
            if isinstance(societal, dict):
                for method in ["human_capital", "friction_cost"]:
                    if (
                        method in societal
                        and "dcea_equity_analysis" in societal[method]
                    ):
                        dcea = societal[method]["dcea_equity_analysis"]
                        interventions.append(f"{name}\n({method})")
                        atkinson_indices.append(dcea.get("atkinson_index", 0))
                        gini_coefficients.append(dcea.get("gini_coefficient", 0))
                        weighted_nmb.append(dcea.get("weighted_total_health_gain", 0))
                        break  # Only take first available method

    if not interventions:
        # No equity data available
        fig.text(
            0.5,
            0.5,
            "No equity analysis data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.tight_layout()
        save_figure(fig, output_dir, "equity_comparison")
        plt.close(fig)
        return

    x = np.arange(len(interventions))

    # Plot 1: Inequality indices
    ax1.bar(
        x - 0.2,
        atkinson_indices,
        width=0.4,
        label="Atkinson Index",
        color="coral",
        alpha=0.8,
        edgecolor="black",
    )
    ax1.bar(
        x + 0.2,
        gini_coefficients,
        width=0.4,
        label="Gini Coefficient",
        color="skyblue",
        alpha=0.8,
        edgecolor="black",
    )
    ax1.set_xlabel("Intervention", fontsize=11)
    ax1.set_ylabel("Inequality Index", fontsize=11)
    ax1.set_title(
        "Inequality Measures\n(Lower = More Equitable)", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(interventions, rotation=45, ha="right", fontsize=9)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, max(max(atkinson_indices), max(gini_coefficients)) * 1.2)

    # Plot 2: Equity-weighted NMB
    colors = plt.cm.viridis(np.linspace(0, 1, len(interventions)))
    bars = ax2.bar(
        x, weighted_nmb, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax2.set_xlabel("Intervention", fontsize=11)
    ax2.set_ylabel("Equity-Weighted Total Health Gain ($)", fontsize=11)
    ax2.set_title(
        "Equity-Weighted Net Monetary Benefit\n(Higher = Better)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(interventions, rotation=45, ha="right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${height / 1e6:.1f}M" if abs(height) >= 1e6 else f"${height / 1e3:.0f}K",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    save_figure(fig, output_dir, "equity_comparison")
    plt.close(fig)


def plot_comprehensive_intervention_summary(
    intervention_results: dict[str, dict],
    bia_results: dict[str, pd.DataFrame],
    wtp_threshold: float = 50000,
    output_dir: str = "output/figures/",
):
    """
    4-panel comprehensive comparison dashboard.

    Args:
        intervention_results: CEA/DCEA results
        bia_results: BIA results
        wtp_threshold: WTP threshold
        output_dir: Output directory
    """
    from .visualizations import apply_default_style, save_figure

    apply_default_style()
    fig = plt.figure(figsize=(16, 12), dpi=1200)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Budget impact
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(bia_results)))
    for idx, (name, bia_df) in enumerate(bia_results.items()):
        years = bia_df["year"].values
        net_costs = bia_df.get("discounted_net_cost", bia_df["net_cost"]).values
        ax1.plot(
            years,
            net_costs,
            "o-",
            linewidth=2,
            markersize=6,
            label=name,
            color=colors[idx],
            alpha=0.8,
        )
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Net Budget Impact ($)")
    ax1.set_title("Budget Impact Comparison", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: NMB comparison
    ax2 = fig.add_subplot(gs[0, 1])
    interventions = list(intervention_results.keys())
    nmb_values = []
    for name in interventions:
        res = intervention_results[name]
        if "societal" in res and isinstance(res["societal"], dict):
            nmb = res["societal"].get("human_capital", {}).get("incremental_nmb", 0)
        else:
            nmb = res.get("incremental_nmb", 0)
        nmb_values.append(nmb)

    colors_nmb = ["g" if nmb > 0 else "r" for nmb in nmb_values]
    ax2.barh(interventions, nmb_values, color=colors_nmb, alpha=0.7, edgecolor="black")
    ax2.axvline(x=0, color="k", linestyle="-", linewidth=1)
    ax2.set_xlabel(f"NMB @ ${wtp_threshold:,}/QALY")
    ax2.set_title("Net Monetary Benefit", fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    # Panel 3: ICER scatter
    ax3 = fig.add_subplot(gs[1, 0])
    inc_qalys = []
    inc_costs = []
    names = []
    for name in interventions:
        res = intervention_results[name]
        if "societal" in res and isinstance(res["societal"], dict):
            data = res["societal"].get("human_capital", {})
        else:
            data = res
        inc_qalys.append(data.get("incremental_qalys", 0))
        inc_costs.append(data.get("incremental_cost", 0))
        names.append(name)

    scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (name, qaly, cost) in enumerate(zip(names, inc_qalys, inc_costs)):
        ax3.scatter(
            qaly,
            cost,
            s=200,
            color=scatter_colors[i],
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
            label=name,
        )

    max_qaly = max(inc_qalys) if inc_qalys else 1
    ax3.plot(
        [0, max_qaly],
        [0, max_qaly * wtp_threshold],
        "r--",
        linewidth=2,
        alpha=0.6,
        label=f"WTP=${wtp_threshold:,}",
    )
    ax3.set_xlabel("Incremental QALYs")
    ax3.set_ylabel("Incremental Cost ($)")
    ax3.set_title("Cost-Effectiveness Plane", fontweight="bold")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    table_data = []
    for name in interventions:
        res = intervention_results[name]
        if "societal" in res and isinstance(res["societal"], dict):
            data = res["societal"].get("human_capital", {})
        else:
            data = res

        icer = data.get("icer", np.inf)
        icer_str = (
            f"${icer:,.0f}"
            if icer not in [np.inf, -np.inf]
            else "Dominated"
            if icer == np.inf
            else "Dominant"
        )

        table_data.append(
            [
                name[:20],  # Truncate long names
                f"${data.get('incremental_cost', 0):,.0f}",
                f"{data.get('incremental_qalys', 0):.2f}",
                icer_str,
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Intervention", "ΔCost", "ΔQALY", "ICER"],
        cellLoc="left",
        loc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax4.set_title("Summary Table", fontweight="bold", pad=20)

    plt.suptitle(
        "Comprehensive Intervention Comparison Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    save_figure(fig, output_dir, "comprehensive_comparison_dashboard")
    plt.close(fig)
