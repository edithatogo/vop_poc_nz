"""
Unified Visualization Module

This module centralizes all plotting functions for the health economic analysis project.
It restores the full plotting suite after refactoring while keeping a single import surface.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


from .value_of_information import ProbabilisticSensitivityAnalysis, calculate_evpi

try:
    from plotnine import (
        aes,
        geom_bar,
        geom_errorbar,
        geom_line,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        theme_bw,
        theme_minimal,
    )

    PLOTNINE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PLOTNINE_AVAILABLE = False

# Shared plotting helpers and defaults
DEFAULT_DPI = 1200
DEFAULT_FORMATS = ("png", "pdf", "svg")


def apply_default_style():
    """Set a consistent matplotlib style for publication-quality figures."""
    preferred_styles = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]
    for st in preferred_styles:
        if st in plt.style.available:
            plt.style.use(st)
            break
    else:  # pragma: no cover - fallback style rarely hit in tests
        plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": DEFAULT_DPI,
            "font.family": "DejaVu Sans",
        }
    )


def slugify(text: str) -> str:
    """Lightweight slugify for filenames."""
    return text.lower().replace(" ", "_")


def save_figure(
    fig,
    output_dir: str,
    filename_base: str,
    dpi: int = DEFAULT_DPI,
    formats: tuple = DEFAULT_FORMATS,
):
    """Save a figure in multiple formats with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f"{filename_base}.{fmt}")
        if fmt == "png":
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_filename_base(
    plot: str,
    intervention: Optional[str] = None,
    perspective: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """Construct a systematic filename base."""
    parts = [slugify(plot)]
    if intervention:
        parts.append(slugify(intervention))
    if perspective:
        parts.append(slugify(perspective))
    if extra:
        parts.append(slugify(extra))
    return "_".join(parts)


def plot_decision_tree(  # pragma: no cover - requires graphviz rendering
    model_name: str, params: Dict, output_dir: str = "output/figures/"
):
    """
    Generates a decision tree diagram for a given intervention.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not GRAPHVIZ_AVAILABLE:
        print("Graphviz not available. Skipping decision tree plot.")
        return

    dot = Digraph(comment=f"Decision Tree for {model_name}")

    # Nodes
    dot.node("D", "Decision", shape="box")
    dot.node("SC", "Standard Care")
    dot.node("NT", "New Treatment")

    # Edges from Decision
    dot.edge("D", "SC")
    dot.edge("D", "NT")

    # States for Standard Care
    for i, state in enumerate(params["states"]):
        dot.node(
            f"SC_{state}",
            f"{state}\nCost: {params['costs']['health_system']['standard_care'][i]}\nQALY: {params['qalys']['standard_care'][i]}",
        )
        dot.edge("SC", f"SC_{state}")

    # States for New Treatment
    for i, state in enumerate(params["states"]):
        dot.node(
            f"NT_{state}",
            f"{state}\nCost: {params['costs']['health_system']['new_treatment'][i]}\nQALY: {params['qalys']['new_treatment'][i]}",
        )
        dot.edge("NT", f"NT_{state}")

    # Transition probabilities
    for i, from_state in enumerate(params["states"]):
        for j, to_state in enumerate(params["states"]):
            prob_sc = params["transition_matrices"]["standard_care"][i][j]
            if prob_sc > 0:
                dot.edge(f"SC_{from_state}", f"SC_{to_state}", label=str(prob_sc))

            prob_nt = params["transition_matrices"]["new_treatment"][i][j]
            if prob_nt > 0:
                dot.edge(f"NT_{from_state}", f"NT_{to_state}", label=str(prob_nt))

    # Render and save the diagram
    base = f"{output_dir}/decision_tree_{slugify(model_name)}"
    dot.render(base, view=False, format="png")
    dot.render(base, view=False, format="pdf")
    dot.render(base, view=False, format="svg")


def plot_cost_effectiveness_plane(
    all_results, output_dir="output/figures/", perspective: Optional[str] = None
):
    """
    Create separate cost-effectiveness plane plot.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Plane: Comparison Across Interventions\n(WTP = $50,000/QALY, 2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    for model_name in model_names:
        data = all_results[model_name]
        ax.scatter(
            data["inc_cost"], data["inc_qaly"], alpha=0.5, s=10, label=model_name
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Incremental Costs ($)", fontsize=12)
    ax.set_ylabel("Incremental QALYs", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("cost_effectiveness_plane", perspective=perspective),
    )


def plot_comparative_ce_plane(
    all_results, output_dir="output/figures/", psa_results: Optional[Dict] = None
):
    """
    Create side-by-side cost-effectiveness plane plots (Health System vs Societal).
    """
    apply_default_style()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Plane: Perspective Comparison\n(WTP = $50,000/QALY)",
        fontsize=16,
        fontweight="bold",
    )

    perspectives = ["health_system", "societal"]
    titles = ["Health System Perspective", "Societal Perspective"]

    for i, perspective in enumerate(perspectives):
        ax = axes[i]

        # Iterate over all interventions
        for model_name, results in all_results.items():
            # Extract data for this perspective
            # Extract data for this perspective
            data = None
            scatter_data = None
            
            if perspective == "health_system":
                # Check for PSA results first (for scatter plot)
                if psa_results and model_name in psa_results:
                    psa_df = psa_results[model_name]
                    if "inc_cost_hs" in psa_df.columns:
                        scatter_data = {
                            "inc_cost": psa_df["inc_cost_hs"],
                            "inc_qaly": psa_df["inc_qaly_hs"]
                        }
                
                # Fallback to deterministic
                if scatter_data is None and "health_system" in results:
                    data = results["health_system"]
                    
            else:  # societal
                # Check for PSA results first (for scatter plot)
                if psa_results and model_name in psa_results:
                    psa_df = psa_results[model_name]
                    if "inc_cost_soc" in psa_df.columns:
                        scatter_data = {
                            "inc_cost": psa_df["inc_cost_soc"],
                            "inc_qaly": psa_df["inc_qaly_soc"]
                        }
                    # Backward compatibility if _soc columns missing but inc_cost exists (legacy)
                    elif "inc_cost" in psa_df.columns:
                         scatter_data = {
                            "inc_cost": psa_df["inc_cost"],
                            "inc_qaly": psa_df["inc_qaly"]
                        }
                
                # Fallback to deterministic
                if scatter_data is None and "societal" in results:
                    soc_res = results["societal"]
                    if "human_capital" in soc_res:
                        data = soc_res["human_capital"]
                    elif "friction_cost" in soc_res:
                        data = soc_res["friction_cost"]

            # Plot Scatter (PSA)
            if scatter_data is not None:
                ax.scatter(
                    scatter_data["inc_cost"],
                    scatter_data["inc_qaly"],
                    alpha=0.4,
                    s=15,
                    label=model_name,
                )
            # Plot Point (Deterministic)
            elif data is not None:
                ax.scatter(
                    data.get("incremental_cost", 0),
                    data.get("incremental_qalys", 0),
                    s=100,
                    marker="D",
                    label=model_name,
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Incremental Costs ($)", fontsize=12)
        ax.set_ylabel("Incremental QALYs", fontsize=12)
        ax.set_title(titles[i], fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        "cost_effectiveness_plane_comparative",
    )


def plot_comparative_ceac(
    psa_results: Dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """
    Plot Comparative Cost-Effectiveness Acceptability Curves (CEAC).
    
    Shows both Health System and Societal perspectives side-by-side.
    """
    output_dir = Path(output_dir)
    wtp_values = np.arange(0, 100001, 1000)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    
    # Define styles for distinct lines to handle overlaps
    # Strategy: Vary width and style so underlying lines are visible
    styles = {
        "HPV Vaccination": {"color": "#1f77b4", "linestyle": "-", "linewidth": 4, "alpha": 0.4, "label": "HPV Vaccination"},
        "Smoking Cessation": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.5, "alpha": 0.9, "label": "Smoking Cessation"},
        "Hepatitis C Therapy": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2.5, "alpha": 0.9, "label": "Hepatitis C Therapy"},
        "Childhood Obesity Prevention": {"color": "#d62728", "linestyle": ":", "linewidth": 2.5, "alpha": 0.9, "label": "Childhood Obesity"},
        "Housing Insulation": {"color": "#9467bd", "linestyle": "-", "linewidth": 1.5, "marker": "o", "markevery": 10, "markersize": 5, "alpha": 0.9, "label": "Housing Insulation"},
    }
    # Fallback style for others
    default_style = {"linestyle": "-", "linewidth": 1, "alpha": 0.7}
    
    # Helper to calculate and plot CEAC for a perspective
    def plot_perspective(ax, perspective_suffix, title):
        print(f"Plotting {title}...")
        for model_name, df in psa_results.items():
            style = styles.get(model_name, default_style.copy())
            if model_name not in styles:
                style["label"] = model_name
            
            # Determine correct columns based on suffix
            if perspective_suffix == "_hs":
                cols = {"q_nt": "qaly_nt_hs", "q_sc": "qaly_sc_hs", "c_nt": "cost_nt_hs", "c_sc": "cost_sc_hs"}
            else: # _soc
                cols = {"q_nt": "qaly_nt_soc", "q_sc": "qaly_sc_soc", "c_nt": "cost_nt_soc", "c_sc": "cost_sc_soc"}
                
            # Check if columns exist (fallback to legacy if needed)
            if cols["q_nt"] not in df.columns:
                if perspective_suffix == "_soc" and "qaly_nt" in df.columns:
                    # Legacy fallback for societal
                    cols = {"q_nt": "qaly_nt", "q_sc": "qaly_sc", "c_nt": "cost_nt", "c_sc": "cost_sc"}
                else:
                    print(f"Skipping {model_name} for {title} (missing columns)")
                    continue

            # Calculate probabilities
            probs = []
            inc_q = df[cols["q_nt"]] - df[cols["q_sc"]]
            inc_c = df[cols["c_nt"]] - df[cols["c_sc"]]
            
            for wtp in wtp_values:
                nmb = inc_q * wtp - inc_c
                prob = np.mean(nmb > 0)
                probs.append(prob)
            
            print(f"  {model_name}: Max Prob = {max(probs):.2%}")
            
            ax.plot(
                wtp_values, 
                probs, 
                **style
            )
        
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Willingness-to-Pay Threshold ($/QALY)", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add 50k line
        ax.axvline(50000, color="gray", linestyle=":", alpha=0.5)
        ax.text(51000, 0.05, "NZ Threshold ($50k)", fontsize=8, color="gray", rotation=90)

    # Plot Health System Perspective
    plot_perspective(ax1, "_hs", "Health System Perspective")
    ax1.set_ylabel("Probability Cost-Effective", fontsize=12)
    
    # Plot Societal Perspective
    plot_perspective(ax2, "_soc", "Societal Perspective")
    
    # Add single legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Make room for legend
    
    save_path = output_dir / "ceac_comparative.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ceac(
    all_results,
    wtp_thresholds,
    output_dir="output/figures/",
    perspective: Optional[str] = None,
):
    """
    Create separate cost-effectiveness acceptability curve plot.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Acceptability Curve: Comparison Across Interventions\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    if not all_results:  # pragma: no cover - nothing to plot
        return

    model_names = list(all_results.keys())
    for model_name in model_names:
        psa_results = all_results[
            model_name
        ]  # This is the DataFrame from psa.run_psa()

        # Calculate CEAC for the societal perspective (as psa_run_cea_wrapper is currently set to societal)
        psa_calculator = ProbabilisticSensitivityAnalysis(
            model_func=lambda params, intervention_type=None: (0, 0),
            parameters={
                "dummy": {"distribution": "normal", "params": {"mean": 0, "std": 1}}
            },
            wtp_threshold=50000,
        )
        ceac_df = psa_calculator.calculate_ceac(psa_results, wtp_values=wtp_thresholds)

        ax.plot(
            ceac_df["wtp_threshold"],
            ceac_df["probability_cost_effective"],
            label=f"{model_name} - Societal",
            linewidth=2,
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("Probability of Cost-Effectiveness", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base(
            "cost_effectiveness_acceptability_curve", perspective=perspective
        ),
    )


def plot_ceaf(
    all_results,
    wtp_thresholds,
    output_dir="output/figures/",
    perspective: Optional[str] = None,
):
    """
    Create separate cost-effectiveness acceptability frontier plot.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Acceptability Frontier: Comparison Across Interventions\n(WTP = $50,000/QALY, 2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    if not all_results:  # pragma: no cover - nothing to plot
        return

    model_names = list(all_results.keys())
    ceac_data_list = []
    for model_name in model_names:
        psa_results = all_results[model_name]
        psa_calculator = ProbabilisticSensitivityAnalysis(
            model_func=lambda params, intervention_type=None: (0, 0),
            parameters={
                "dummy": {"distribution": "normal", "params": {"mean": 0, "std": 1}}
            },
            wtp_threshold=50000,
        )
        ceac_df = psa_calculator.calculate_ceac(psa_results, wtp_values=wtp_thresholds)
        ceac_data_list.append(ceac_df["probability_cost_effective"])

        ax.plot(
            ceac_df["wtp_threshold"],
            ceac_df["probability_cost_effective"],
            label=f"{model_name} - Societal",
            linewidth=2,
        )

    # Calculate frontier: max CE probability across models at each WTP
    if ceac_data_list:
        frontier = np.maximum.reduce(ceac_data_list)
        ax.plot(
            wtp_thresholds,
            frontier,
            label="Frontier",
            color="red",
            linewidth=3,
            linestyle="-.",
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("Probability of Cost-Effectiveness", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base(
            "cost_effectiveness_acceptability_frontier", perspective=perspective
        ),
    )


def plot_evpi(
    all_results,
    wtp_thresholds,
    output_dir="output/figures/",
    perspective: Optional[str] = None,
):
    """
    Create separate expected value of perfect information plot.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Expected Value of Perfect Information: Comparison Across Interventions\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    for model_name in model_names:
        psa_results = all_results[model_name]  # This is the DataFrame from psa.run_sa()

        # Calculate EVPI for each WTP threshold
        evpi_values = [
            calculate_evpi(psa_results, wtp_threshold=wtp) for wtp in wtp_thresholds
        ]

        ax.plot(
            wtp_thresholds,
            evpi_values,
            label=f"{model_name} - Societal",
            linewidth=2,
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("Expected Value of Perfect Information ($)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base(
            "expected_value_perfect_information", perspective=perspective
        ),
    )


def plot_comparative_evpi(
    all_results,
    wtp_thresholds,
    output_dir="output/figures/",
):
    """
    Create side-by-side Expected Value of Perfect Information (EVPI) plots.
    """
    apply_default_style()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    fig.suptitle(
        "Expected Value of Perfect Information: Perspective Comparison\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )

    perspectives = ["health_system", "societal"]
    titles = ["Health System Perspective", "Societal Perspective"]

    for i, perspective in enumerate(perspectives):
        ax = axes[i]
        
        # Iterate over all interventions
        for model_name, psa_results in all_results.items():
            # Prepare DataFrame with correct columns for calculate_evpi
            df_calc = psa_results.copy()
            
            if perspective == "health_system":
                if "qaly_sc_hs" in df_calc.columns:
                    df_calc["qaly_sc"] = df_calc["qaly_sc_hs"]
                    df_calc["cost_sc"] = df_calc["cost_sc_hs"]
                    df_calc["qaly_nt"] = df_calc["qaly_nt_hs"]
                    df_calc["cost_nt"] = df_calc["cost_nt_hs"]
                else:
                    print(f"Skipping {model_name} for HS EVPI (missing columns)")
                    continue
            else: # societal
                if "qaly_sc_soc" in df_calc.columns:
                    df_calc["qaly_sc"] = df_calc["qaly_sc_soc"]
                    df_calc["cost_sc"] = df_calc["cost_sc_soc"]
                    df_calc["qaly_nt"] = df_calc["qaly_nt_soc"]
                    df_calc["cost_nt"] = df_calc["cost_nt_soc"]
                # Else rely on default aliasing in psa_results if _soc columns missing
            
            # Calculate EVPI for each WTP threshold
            evpi_values = [
                calculate_evpi(df_calc, wtp_threshold=wtp) for wtp in wtp_thresholds
            ]
            
            # Check if all zero
            if all(v == 0 for v in evpi_values):
                print(f"  {model_name} ({perspective}): EVPI is all zero.")
            
            ax.plot(
                wtp_thresholds,
                evpi_values,
                label=model_name,
                linewidth=2,
                alpha=0.8
            )

        ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
        ax.set_ylabel("Expected Value of Perfect Information ($)", fontsize=12)
        ax.set_title(titles[i], fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend to first plot only or both? Both is safer if interventions differ
        # But let's put a single legend at bottom
        # ax.legend(fontsize=10)

    # Add single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Make room for legend

    save_figure(
        fig,
        output_dir,
        "expected_value_perfect_information_comparative",
    )


def plot_net_benefit_curves(
    all_results,
    wtp_thresholds,
    output_dir="output/figures/",
    perspective: Optional[str] = None,
):
    """
    Create net benefit curves with confidence intervals.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)  # Only one subplot for societal
    fig.suptitle(
        "Net Benefit Curves: Societal Perspective Across Interventions\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())

    # Societal Perspective
    for model_name in model_names:
        psa_results = all_results[model_name]
        nmb_mean = []
        nmb_lower = []
        nmb_upper = []

        for wtp in wtp_thresholds:
            # Calculate incremental NMB for each PSA iteration at the current WTP
            nmb_sc_iter = (psa_results["qaly_sc"] * wtp) - psa_results["cost_sc"]
            nmb_nt_iter = (psa_results["qaly_nt"] * wtp) - psa_results["cost_nt"]
            inc_nmb_iter = nmb_nt_iter - nmb_sc_iter

            nmb_mean.append(np.mean(inc_nmb_iter))
            nmb_lower.append(np.percentile(inc_nmb_iter, 2.5))
            nmb_upper.append(np.percentile(inc_nmb_iter, 97.5))

        ax.plot(wtp_thresholds, nmb_mean, label=f"{model_name} - Mean", linewidth=2)
        ax.fill_between(
            wtp_thresholds,
            nmb_lower,
            nmb_upper,
            alpha=0.3,
            label=f"{model_name} - 95% CI",
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("Net Monetary Benefit ($)", fontsize=12)
    ax.set_title("Societal Perspective")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("net_benefit_curves", perspective=perspective),
    )


def plot_value_of_perspective(
    all_results,
    wtp_thresholds,
    output_dir="output/figures/",
    perspective: Optional[str] = None,
):
    """
    Create a plot showing probability of cost-effectiveness from societal perspective.
    Temporarily simplified as only societal perspective PSA is currently available.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Probability of Cost-Effectiveness (Societal Perspective) at WTP = $50,000/QALY\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    wtp_50k_idx = np.argmin(np.abs(np.array(wtp_thresholds) - 50000))
    wtp_50k = wtp_thresholds[wtp_50k_idx]

    prob_cost_effective_at_50k = []

    for model_name in model_names:
        psa_results = all_results[model_name]

        # Calculate NMB for the societal perspective at WTP = $50,000
        nmb_sc_50k = (psa_results["qaly_sc"] * wtp_50k) - psa_results["cost_sc"]
        nmb_nt_50k = (psa_results["qaly_nt"] * wtp_50k) - psa_results["cost_nt"]
        inc_nmb_50k = nmb_nt_50k - nmb_sc_50k

        prob_ce = np.mean(inc_nmb_50k > 0)
        prob_cost_effective_at_50k.append(prob_ce)

    ax.bar(
        model_names,
        prob_cost_effective_at_50k,
        color=["blue", "green", "red"],
        alpha=0.7,
    )
    ax.set_ylabel("Probability of Cost-Effectiveness", fontsize=12)
    ax.set_title(
        f"Probability of Cost-Effective Decision at WTP = ${wtp_50k:,.0f}/QALY"
    )
    ax.set_ylim(0, 1)  # Probability is between 0 and 1
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("value_of_perspective", perspective=perspective),
    )


def plot_pop_evpi(
    all_results,
    wtp_thresholds,
    population_sizes: Dict[str, int],
    output_dir="output/figures/",
    perspective: Optional[str] = None,
):
    """
    Create population EVPI plots for the societal perspective.
    """
    apply_default_style()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Population Expected Value of Perfect Information (Societal Perspective)\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    for model_name in model_names:
        psa_results = all_results[model_name]
        # Use provided population size, default to 1 if missing (with warning ideally)
        pop_size = population_sizes.get(model_name, 1)

        # Calculate EVPI per person for each WTP threshold
        evpi_per_person_values = [
            calculate_evpi(psa_results, wtp_threshold=wtp) for wtp in wtp_thresholds
        ]

        # Calculate population EVPI
        pop_evpi_values = np.array(evpi_per_person_values) * pop_size

        ax.plot(
            wtp_thresholds,
            pop_evpi_values,
            label=f"{model_name} - Societal",
            linewidth=2,
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("Population EVPI ($)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    save_figure(
        fig, output_dir, build_filename_base("population_evpi", perspective=perspective)
    )


def plot_evppi(voi_report, output_dir="output/figures/"):
    """
    Plot Expected Value of Partial Perfect Information curves.
    """
    apply_default_style()

    evppi_data = voi_report["value_of_information"]["evppi_by_parameter_group"]
    wtp_thresholds = voi_report["value_of_information"]["wtp_thresholds"]

    if not evppi_data:  # pragma: no cover - nothing to plot
        print("No EVPPI data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Expected Value of Partial Perfect Information (Societal Perspective)\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    for param_group, evppi_values in evppi_data.items():
        ax.plot(
            wtp_thresholds,
            evppi_values,
            label=param_group.replace("EVPPI_", "").replace("_", " ").title(),
            linewidth=2,
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("EVPPI ($)", fontsize=12)
    ax.set_title("EVPPI by Parameter Group (Societal Perspective)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base(
            "expected_value_partial_perfect_information", perspective="societal"
        ),
    )


def plot_comparative_evppi(voi_results, output_dir="output/figures/"):
    """
    Plot Comparative Expected Value of Partial Perfect Information (EVPPI).
    
    Creates a side-by-side comparison of EVPPI for Health System and Societal perspectives.
    Plots all interventions on each subplot.
    """
    apply_default_style()
    
    if not voi_results:
        print("No VOI results to plot.")
        return

    # Collect all data into a long-form DataFrame for plotting
    data_rows = []
    
    for model_name, report in voi_results.items():
        evppi_data = report["value_of_information"]["evppi_by_parameter_group"]
        wtp_thresholds = report["value_of_information"]["wtp_thresholds"]
        
        for key, values in evppi_data.items():
            # Key format: EVPPI_{GroupName}_{Suffix}
            # Suffix is _HS or _Soc
            if key.endswith("_HS"):
                perspective = "Health System"
                group_name = key.replace("EVPPI_", "").replace("_HS", "").replace("_", " ")
            elif key.endswith("_Soc"):
                perspective = "Societal"
                group_name = key.replace("EVPPI_", "").replace("_Soc", "").replace("_", " ")
            else:
                # Fallback for old keys or generic ones
                perspective = "Societal"
                group_name = key.replace("EVPPI_", "").replace("_", " ")

            # Filter for key groups to avoid clutter
            # We want to show "Cost Parameters" and "QALY Parameters" mainly
            # Or specific ones like "Health System Costs"
            
            # Let's simplify: Show "Cost Parameters" and "QALY Parameters" for each intervention
            if "Cost Parameters" not in group_name and "QALY Parameters" not in group_name:
                continue

            for wtp, val in zip(wtp_thresholds, values):
                data_rows.append({
                    "Intervention": model_name,
                    "Perspective": perspective,
                    "Parameter Group": group_name,
                    "WTP": wtp,
                    "EVPPI": val
                })
                
    if not data_rows:
        print("No EVPPI data rows extracted.")
        return
        
    df = pd.DataFrame(data_rows)
    
    # Plotting: 2 Subplots (Health System, Societal)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300, sharey=True)
    
    fig.suptitle(
        "Comparative Expected Value of Partial Perfect Information (EVPPI)\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )
    
    perspectives = ["Health System", "Societal"]
    axes = [ax1, ax2]
    
    # Define line styles for parameter groups
    # Solid for Cost, Dashed for QALY
    linestyles = {"Cost Parameters": "-", "QALY Parameters": "--"}
    
    # Define colors for interventions (consistent with other plots if possible)
    # We'll use the default color cycle
    interventions = df["Intervention"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(interventions)))
    color_map = dict(zip(interventions, colors))
    
    for ax, perspective in zip(axes, perspectives):
        subset = df[df["Perspective"] == perspective]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(f"{perspective} Perspective")
            continue
            
        for intervention in interventions:
            int_data = subset[subset["Intervention"] == intervention]
            if int_data.empty:
                continue
                
            c = color_map[intervention]
            
            for group in ["Cost Parameters", "QALY Parameters"]:
                group_data = int_data[int_data["Parameter Group"] == group]
                if not group_data.empty:
                    ax.plot(
                        group_data["WTP"], 
                        group_data["EVPPI"], 
                        label=f"{intervention} ({group})", 
                        color=c,
                        linestyle=linestyles.get(group, "-"),
                        linewidth=2
                    )
        
        ax.set_title(f"{perspective} Perspective", fontsize=14, fontweight="bold")
        ax.set_xlabel("WTP Threshold ($/QALY)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format axis labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}k"))
        
    ax1.set_ylabel("EVPPI ($)", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    # Create a custom legend
    # We want to separate Intervention (Color) and Parameter Group (Line Style)
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    
    legend_elements = []
    # Interventions
    # Use an invisible patch for the title
    legend_elements.append(mpatches.Patch(color='none', label=r'$\bf{Interventions:}$'))
    for intervention in interventions:
        legend_elements.append(Line2D([0], [0], color=color_map[intervention], lw=2, label=intervention))
    
    # Spacing
    legend_elements.append(mpatches.Patch(color='none', label=' '))
    
    # Parameter Groups
    legend_elements.append(mpatches.Patch(color='none', label=r'$\bf{Parameters:}$'))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', label='Cost Parameters'))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label='QALY Parameters'))
    
    # Place legend below the plots
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle and legend
    save_figure(
        fig,
        output_dir,
        "expected_value_partial_perfect_information_comparative",
    )


def plot_comparative_ce_plane_with_delta(all_results, output_dir="output/figures/"):
    """
    Create comparative Cost-Effectiveness Plane with Delta subplot.
    3 Subplots: Health System, Societal, Delta (Societal - HS).
    """
    apply_default_style()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(
        "Comparative Cost-Effectiveness Plane with Perspective Delta\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )

    perspectives = ["Health System", "Societal", "Delta (Societal - HS)"]
    
    # Define colors for interventions
    model_names = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = dict(zip(model_names, colors))

    for model_name in model_names:
        psa_results = all_results[model_name]
        c = color_map[model_name]
        
        # Extract data
        # Health System
        inc_cost_hs = psa_results["inc_cost_hs"]
        inc_qaly_hs = psa_results["inc_qaly_hs"]
        
        # Societal
        inc_cost_soc = psa_results["inc_cost_soc"]
        inc_qaly_soc = psa_results["inc_qaly_soc"]
        
        # Delta
        delta_cost = inc_cost_soc - inc_cost_hs
        delta_qaly = inc_qaly_soc - inc_qaly_hs
        
        # Plot Health System
        axes[0].scatter(
            inc_qaly_hs,
            inc_cost_hs,
            label=model_name,
            color=c,
            alpha=0.15,
            s=10,
            edgecolors="none"
        )
        # Plot Mean Point
        axes[0].scatter(
            np.mean(inc_qaly_hs),
            np.mean(inc_cost_hs),
            color=c,
            s=100,
            edgecolors="black",
            marker="o"
        )

        # Plot Societal
        axes[1].scatter(
            inc_qaly_soc,
            inc_cost_soc,
            label=model_name,
            color=c,
            alpha=0.15,
            s=10,
            edgecolors="none"
        )
        axes[1].scatter(
            np.mean(inc_qaly_soc),
            np.mean(inc_cost_soc),
            color=c,
            s=100,
            edgecolors="black",
            marker="o"
        )
        
        # Plot Delta
        axes[2].scatter(
            delta_qaly,
            delta_cost,
            label=model_name,
            color=c,
            alpha=0.15,
            s=10,
            edgecolors="none"
        )
        axes[2].scatter(
            np.mean(delta_qaly),
            np.mean(delta_cost),
            color=c,
            s=100,
            edgecolors="black",
            marker="o"
        )

    # Formatting
    for i, ax in enumerate(axes):
        ax.set_title(perspectives[i], fontsize=14, fontweight="bold")
        ax.set_xlabel("Incremental QALYs", fontsize=12)
        ax.set_ylabel("Incremental Cost ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        
        # WTP Threshold line (50k)
        # y = 50000 * x
        xlim = ax.get_xlim()
        x_vals = np.array(xlim)
        y_vals = 50000 * x_vals
        ax.plot(x_vals, y_vals, 'k--', alpha=0.5, label="WTP $50k/QALY")
        ax.set_xlim(xlim) # Restore limits
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:,.0f}k"))

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Filter for unique labels (interventions + WTP)
    by_label = dict(zip(labels, handles))
    # We only want intervention labels, WTP is common
    
    # Create custom legend for interventions
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color_map[name], markersize=10) for name in model_names]
    legend_elements.append(Line2D([0], [0], color='k', linestyle='--', label='WTP $50k/QALY'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(model_names)+1, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(
        fig,
        output_dir,
        "cost_effectiveness_plane_comparative_with_delta",
    )


def plot_comparative_two_way_dsa(comparative_results, output_dir="output/figures/"):
    """
    Create plots for comparative two-way DSA results.
    """
    apply_default_style()

    for comparison_name, data in comparative_results.items():
        print(f"Creating comparative plots for {comparison_name}...")

        # Create subplots for different perspectives and comparisons
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle(
            f"Comparative Two-Way DSA: {comparison_name}\n(WTP = $50,000/QALY)",
            fontsize=16,
            fontweight="bold",
        )

        perspectives = ["hs", "soc"]
        comparison_pairs = ["inc_nmb"]  # Use the actual key from comparative_grid

        for i, perspective in enumerate(perspectives):
            for j, pair in enumerate(comparison_pairs):
                ax = axes[i, j]

                # Extract data for this comparison
                nmb_key = f"{pair}_{perspective}"
                nmb_values = [item[nmb_key] for item in data["comparative_grid"]]

                # Create a simple heatmap-like visualization
                # For simplicity, we'll show the distribution of NMB values
                ax.hist(nmb_values, bins=30, alpha=0.7, edgecolor="black")
                ax.axvline(
                    x=0,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Cost-Effective Threshold",
                )
                ax.set_xlabel("Incremental NMB ($)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{comparison_name}\n({perspective.upper()})")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(
            fig,
            output_dir,
            f"comparative_two_way_dsa_{comparison_name.replace(' ', '_').replace('vs', 'vs')}",
        )


def plot_comparative_three_way_dsa(comparative_results, output_dir="output/figures/"):
    """
    Create plots for comparative three-way DSA results.
    """
    apply_default_style()

    for comparison_name, data in comparative_results.items():
        print(f"Creating comparative three-way plots for {comparison_name}...")

        # Create comprehensive visualization showing all pairwise comparisons
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
        fig.suptitle(
            f"Comparative Three-Way DSA: {comparison_name}\n(WTP = $50,000/QALY)",
            fontsize=16,
            fontweight="bold",
        )

        perspectives = ["hs", "soc"]
        comparison_pairs = ["hpv_vs_smoking", "hpv_vs_hep_c", "smoking_vs_hep_c"]

        for i, perspective in enumerate(perspectives):
            for j, pair in enumerate(comparison_pairs):
                ax = axes[i, j]

                # Extract data for this comparison - use the correct key names from comparative_grid
                if pair == "hpv_vs_smoking":
                    nmb_key = f"hpv_vs_smoking_nmb_{perspective}"
                elif pair == "hpv_vs_hep_c":
                    nmb_key = f"hpv_vs_hep_c_nmb_{perspective}"
                else:  # smoking_vs_hep_c
                    nmb_key = f"smoking_vs_hep_c_nmb_{perspective}"

                nmb_values = [item[nmb_key] for item in data["comparative_grid"]]

                # Create histogram
                ax.hist(
                    nmb_values, bins=25, alpha=0.7, edgecolor="black", color=f"C{j}"
                )
                ax.axvline(
                    x=0,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Cost-Effective",
                )
                ax.set_xlabel("Incremental NMB ($)")
                ax.set_ylabel("Frequency")
                ax.set_title(
                    f"{pair.replace('_', ' vs ').title()}\n({perspective.upper()})"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(
            fig,
            output_dir,
            f"comparative_three_way_dsa_{comparison_name.replace(' ', '_').replace('vs', 'vs')}",
        )


def plot_cluster_analysis(cluster_results, output_dir="output/figures/"):
    """
    Create comprehensive cluster analysis visualizations.
    """
    apply_default_style()

    for intervention_name, results in cluster_results.items():
        features_pca = results["features_pca"]
        cluster_labels = results["cluster_labels"]
        n_clusters = results["n_clusters"]
        cluster_analysis = results["cluster_analysis"]

        # Create figure with subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle(
            f"Cluster Analysis Results: {intervention_name}\n({n_clusters} Clusters, Silhouette Score: {results['silhouette_score']:.3f})",
            fontsize=16,
            fontweight="bold",
        )

        # 1. PCA scatter plot
        ax = axes[0, 0]
        colors = ["blue", "red", "green", "orange", "purple"][:n_clusters]
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = cluster_analysis[f"cluster_{cluster_id}"]["size"]
            cluster_pct = cluster_analysis[f"cluster_{cluster_id}"]["percentage"]

            ax.scatter(
                features_pca[cluster_mask, 0],
                features_pca[cluster_mask, 1],
                c=colors[cluster_id],
                alpha=0.6,
                s=30,
                label=f"Cluster {cluster_id} ({cluster_pct:.1f}%, n={cluster_size})",
            )

        # Plot cluster centers
        for cluster_id in range(n_clusters):
            center = results["cluster_centers"][cluster_id]
            ax.scatter(
                center[0],
                center[1],
                c=colors[cluster_id],
                marker="x",
                s=200,
                linewidths=3,
            )

        ax.set_xlabel(
            f"PC1 ({results['pca_explained_variance'][0] * 100:.1f}% variance)"
        )
        ax.set_ylabel(
            f"PC2 ({results['pca_explained_variance'][1] * 100:.1f}% variance)"
        )
        ax.set_title("Cluster Distribution (PCA)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cluster characteristics comparison
        ax = axes[0, 1]
        feature_names = [
            "Inc Cost",
            "Inc QALYs",
            "ICER",
            "NMB $20k",
            "NMB $50k",
            "NMB $100k",
        ]
        cluster_means = []

        for cluster_id in range(n_clusters):
            means = cluster_analysis[f"cluster_{cluster_id}"]["means"][
                :6
            ]  # First 6 features
            cluster_means.append(means)

        cluster_means = np.array(cluster_means).T
        x = np.arange(len(feature_names))

        for cluster_id in range(n_clusters):
            ax.bar(
                x + cluster_id * 0.15,
                cluster_means[:, cluster_id],
                width=0.15,
                label=f"Cluster {cluster_id}",
                color=colors[cluster_id],
                alpha=0.7,
            )

        ax.set_xlabel("Cost-Effectiveness Metrics")
        ax.set_ylabel("Mean Value")
        ax.set_title("Cluster Characteristics Comparison")
        ax.set_xticks(x + 0.15)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cost-effectiveness plane by cluster
        ax = axes[1, 0]
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = results["features"][cluster_mask]

            ax.scatter(
                cluster_features[:, 0],
                cluster_features[:, 1],
                c=colors[cluster_id],
                alpha=0.6,
                s=30,
                label=f"Cluster {cluster_id}",
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Incremental Costs ($)")
        ax.set_ylabel("Incremental QALYs")
        ax.set_title("Cost-Effectiveness Plane by Cluster")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Key differentiators
        ax = axes[1, 1]
        for cluster_id in range(n_clusters):
            distinctive_features = cluster_analysis[f"cluster_{cluster_id}"][
                "distinctive_features"
            ][:3]
            feature_names_distinctive = [
                cluster_analysis[f"cluster_{cluster_id}"]["feature_names"][i]
                for i in range(3)
            ]
            standardized_diffs = cluster_analysis[f"cluster_{cluster_id}"][
                "standardized_differences"
            ][distinctive_features]

            ax.barh(
                [f"C{cluster_id}: {name}" for name in feature_names_distinctive],
                standardized_diffs,
                color=colors[cluster_id],
                alpha=0.7,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Standardized Difference from Overall Mean")
        ax.set_title("Key Differentiators by Cluster")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, output_dir, f"cluster_analysis_{slugify(intervention_name)}")


def plot_comparative_clusters(cluster_results, output_dir="output/figures/"):
    """
    Create comparative cluster analysis across all interventions.
    """
    apply_default_style()

    if not cluster_results:  # pragma: no cover - nothing to plot
        print("No cluster results available for comparison")
        return

    # Create comparative visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    fig.suptitle(
        "Comparative Cluster Analysis Across Interventions",
        fontsize=16,
        fontweight="bold",
    )

    intervention_names = list(cluster_results.keys())
    colors = ["blue", "red", "green"]

    # 1. Cluster sizes comparison
    ax = axes[0, 0]
    all_cluster_sizes = []
    all_cluster_labels = []

    for i, intervention in enumerate(intervention_names):
        results = cluster_results[intervention]
        cluster_sizes = []
        cluster_labels = []

        for cluster_id in range(results["n_clusters"]):
            percentage = results["cluster_analysis"][f"cluster_{cluster_id}"][
                "percentage"
            ]
            cluster_sizes.append(percentage)
            cluster_labels.append(f"C{cluster_id}")

        ax.bar(
            [x + i * 0.25 for x in range(len(cluster_sizes))],
            cluster_sizes,
            width=0.25,
            label=intervention,
            color=colors[i],
            alpha=0.7,
        )

        # Store for axis setup (use the last intervention's cluster count)
        all_cluster_sizes = cluster_sizes
        all_cluster_labels = cluster_labels

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage of Simulations (%)")
    ax.set_title("Cluster Size Distribution")
    ax.set_xticks([x + 0.25 for x in range(len(all_cluster_sizes))])
    ax.set_xticklabels(all_cluster_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Silhouette scores comparison
    ax = axes[0, 1]
    silhouette_scores = [
        cluster_results[name]["silhouette_score"] for name in intervention_names
    ]
    ax.bar(intervention_names, silhouette_scores, color=colors, alpha=0.7)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Quality Comparison")
    ax.grid(True, alpha=0.3)

    # 3. Number of clusters comparison
    ax = axes[0, 2]
    n_clusters_list = [
        cluster_results[name]["n_clusters"] for name in intervention_names
    ]
    ax.bar(intervention_names, n_clusters_list, color=colors, alpha=0.7)
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Optimal Cluster Number")
    ax.grid(True, alpha=0.3)

    # 4. Key metrics comparison across interventions
    ax = axes[1, 0]
    metrics = ["Incremental Cost", "Incremental QALYs", "ICER"]
    for i, intervention in enumerate(intervention_names):
        results = cluster_results[intervention]
        # Use the largest cluster for comparison
        largest_cluster = max(
            range(results["n_clusters"]),
            key=lambda x: results["cluster_analysis"][f"cluster_{x}"]["size"],
        )
        cluster_data = results["cluster_analysis"][f"cluster_{largest_cluster}"]
        means = cluster_data["means"][:3]  # First 3 metrics

        ax.bar(
            [x + i * 0.25 for x in range(len(metrics))],
            means,
            width=0.25,
            label=intervention,
            color=colors[i],
            alpha=0.7,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean Value")
    ax.set_title("Key Metrics by Intervention (Largest Cluster)")
    ax.set_xticks([x + 0.25 for x in range(len(metrics))])
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Cost-effectiveness comparison
    ax = axes[1, 1]
    for i, intervention in enumerate(intervention_names):
        results = cluster_results[intervention]
        # Plot cost-effectiveness points colored by cluster
        features = results["features"]
        cluster_labels = results["cluster_labels"]

        for cluster_id in range(results["n_clusters"]):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            ax.scatter(
                cluster_features[:, 0],
                cluster_features[:, 1],
                c=[colors[i]],
                alpha=0.6,
                s=20,
                marker=["o", "s", "^"][cluster_id % 3],
                label=f"{intervention} C{cluster_id}" if cluster_id == 0 else "",
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Incremental Costs ($)")
    ax.set_ylabel("Incremental QALYs")
    ax.set_title("Comparative Cost-Effectiveness by Cluster")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Archetype identification
    ax = axes[1, 2]
    archetype_data = []

    for intervention in intervention_names:
        results = cluster_results[intervention]
        for cluster_id in range(results["n_clusters"]):
            cluster_data = results["cluster_analysis"][f"cluster_{cluster_id}"]
            icer = cluster_data["means"][2]  # ICER is at index 2
            nmb_50k = cluster_data["means"][4]  # NMB at $50k is at index 4
            archetype_data.append(
                {
                    "intervention": intervention,
                    "cluster": cluster_id,
                    "icer": icer,
                    "nmb_50k": nmb_50k,
                    "size": cluster_data["percentage"],
                }
            )

    # Plot archetypes
    for _i, data in enumerate(archetype_data):
        ax.scatter(
            data["icer"],
            data["nmb_50k"],
            s=data["size"] * 10,  # Size proportional to cluster percentage
            c=colors[intervention_names.index(data["intervention"])],
            alpha=0.7,
            marker=["o", "s", "^"][data["cluster"] % 3],
            label=f"{data['intervention']} C{data['cluster']}",
        )

    ax.set_xlabel("ICER ($/QALY)")
    ax.set_ylabel("NMB at $50,000/QALY ($)")
    ax.set_title("Intervention Archetypes")
    ax.axhline(
        0, color="red", linestyle="--", linewidth=1, label="Cost-Effective Threshold"
    )
    ax.axvline(
        50000, color="orange", linestyle="--", linewidth=1, label="$50k WTP Threshold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "comparative_cluster_analysis")


# ---------------------------------------------------------------------------
# Additional high-value visualizations
# ---------------------------------------------------------------------------


def plot_markov_trace(
    model_results: Dict,
    output_dir: str = "output/figures/",
    intervention_name: str = "Intervention",
):
    """
    Plot the proportion of the cohort in each health state over time.

    Expects model_results to contain a pandas DataFrame under the key 'trace_dataframe'
    (index = cycle, columns = state names, values = proportions or counts).
    """
    trace = model_results.get("trace_dataframe")
    if trace is None or trace.empty:  # pragma: no cover - guard
        print("No trace_dataframe found for Markov trace plot.")
        return

    apply_default_style()
    ax = trace.plot(kind="area", stacked=True, alpha=0.85, figsize=(10, 6))
    ax.set_title(f"Markov Trace: {intervention_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Proportion of Cohort")
    ax.set_xlabel("Cycle")
    ax.legend(title="Health States", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(
        ax.figure, output_dir, build_filename_base("markov_trace", intervention_name)
    )


def plot_cost_qaly_breakdown(
    cost_qaly_data: Dict[str, Dict[str, float]],
    output_dir: str = "output/figures/",
    intervention_name: str = "Intervention",
):
    """
    Stacked bar chart showing cost and QALY components for standard care vs new treatment.

    cost_qaly_data structure:
    {
        "standard_care": {"Drug": ..., "Admin": ..., "AE": ..., "Monitoring": ..., "QALY_LY": ..., "QALY_Utility": ...},
        "new_treatment": {...}
    }
    QALY components are plotted separately from cost components for clarity.
    """
    if not cost_qaly_data:  # pragma: no cover - guard
        print("No cost/QALY breakdown data provided.")
        return

    apply_default_style()
    categories: List[str] = []
    for arm in cost_qaly_data.values():
        categories.extend(arm.keys())
    categories = sorted(set(categories))

    arms = list(cost_qaly_data.keys())
    cost_cats = [c for c in categories if not c.lower().startswith("qaly")]
    qaly_cats = [c for c in categories if c.lower().startswith("qaly")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # Costs
    bottom = np.zeros(len(arms))
    for cat in cost_cats:
        values = [cost_qaly_data.get(arm, {}).get(cat, 0) for arm in arms]
        axes[0].bar(arms, values, bottom=bottom, label=cat)
        bottom += np.array(values)
    axes[0].set_title(f"Cost Breakdown: {intervention_name}")
    axes[0].set_ylabel("Costs ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    # QALYs
    bottom = np.zeros(len(arms))
    for cat in qaly_cats:
        values = [cost_qaly_data.get(arm, {}).get(cat, 0) for arm in arms]
        axes[1].bar(arms, values, bottom=bottom, label=cat)
        bottom += np.array(values)
    axes[1].set_title(f"QALY Breakdown: {intervention_name}")
    axes[1].set_ylabel("QALYs")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(
        fig, output_dir, build_filename_base("cost_qaly_breakdown", intervention_name)
    )


def plot_density_ce_plane(
    inc_costs: List[float],
    inc_qalys: List[float],
    wtp_threshold: float = 50000,
    output_dir: str = "output/figures/",
    use_hexbin: bool = True,
    intervention_label: str = "All",
    use_plotnine: bool = False,
):
    """
    Density visualization for the cost-effectiveness plane using hexbin or KDE.
    """
    if len(inc_costs) == 0 or len(inc_qalys) == 0:  # pragma: no cover - guard
        print("No incremental cost/QALY data provided for density CE plane.")
        return

    if (
        use_plotnine and not PLOTNINE_AVAILABLE
    ):  # pragma: no cover - optional dependency
        print("plotnine not available; falling back to matplotlib.")
        use_plotnine = False

    if use_plotnine:  # pragma: no cover - optional dependency
        df = pd.DataFrame({"inc_cost": inc_costs, "inc_qaly": inc_qalys})
        plt_obj = (
            ggplot(df, aes(x="inc_cost", y="inc_qaly"))
            + geom_point(alpha=0.2, color="steelblue")
            + geom_line(
                aes(x="inc_cost", y=f"inc_cost / {wtp_threshold}"),
                color="green",
                linetype="--",
            )
            + theme_bw()
            + labs(
                title=f"Density Cost-Effectiveness Plane: {intervention_label}",
                x="Incremental Costs ($)",
                y="Incremental QALYs",
            )
        )
        fname = build_filename_base(
            "density_ce_plane", intervention_label, None, "plotnine"
        )
        path_png = os.path.join(output_dir, f"{fname}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt_obj.save(path_png, dpi=DEFAULT_DPI)
    else:
        apply_default_style()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        if use_hexbin:
            hb = ax.hexbin(
                inc_costs, inc_qalys, gridsize=50, cmap="Blues", mincnt=1, alpha=0.9
            )
            fig.colorbar(hb, ax=ax, label="Count")
        else:
            sns.kdeplot(
                x=inc_costs,
                y=inc_qalys,
                fill=True,
                cmap="Blues",
                thresh=0.05,
                levels=15,
                ax=ax,
            )
        ax.scatter(
            np.mean(inc_costs),
            np.mean(inc_qalys),
            c="red",
            marker="x",
            s=60,
            label="Mean",
        )
        x_line = np.linspace(min(inc_costs), max(inc_costs), 100)
        ax.plot(
            x_line,
            x_line / wtp_threshold,
            linestyle="--",
            color="green",
            label=f"WTP = ${wtp_threshold:,.0f}/QALY",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Incremental Costs ($)")
        ax.set_ylabel("Incremental QALYs")
        ax.set_title(f"Density Cost-Effectiveness Plane: {intervention_label}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(
            fig, output_dir, build_filename_base("density_ce_plane", intervention_label)
        )


def plot_subgroup_forest(
    subgroup_results: pd.DataFrame,
    output_dir: str = "output/figures/",
    metric: str = "ICER",
    wtp_threshold: Optional[float] = None,
    use_plotnine: bool = False,
):
    """
    Forest plot for subgroup heterogeneity.

    subgroup_results DataFrame expected columns: ['subgroup', 'estimate', 'ci_lower', 'ci_upper']
    metric can be 'ICER' or 'NMB'. If wtp_threshold provided, draw reference line.
    """
    if subgroup_results is None or subgroup_results.empty:  # pragma: no cover - guard
        print("No subgroup results provided for forest plot.")
        return

    if (
        use_plotnine and not PLOTNINE_AVAILABLE
    ):  # pragma: no cover - optional dependency
        print("plotnine not available; falling back to matplotlib.")
        use_plotnine = False

    if use_plotnine:  # pragma: no cover - optional dependency
        df = subgroup_results.copy()
        df["subgroup_ordered"] = pd.Categorical(
            df["subgroup"],
            categories=df.sort_values("estimate")["subgroup"],
            ordered=True,
        )
        plt_obj = (
            ggplot(df, aes(x="estimate", y="subgroup_ordered"))
            + geom_point(color="black")
            + geom_errorbar(
                aes(xmin="ci_lower", xmax="ci_upper"), color="steelblue", height=0.2
            )
        )
        if wtp_threshold is not None and metric.lower() == "icer":
            plt_obj = plt_obj + geom_vline(
                xintercept=wtp_threshold, linetype="--", color="green"
            )
        plt_obj = plt_obj + geom_vline(xintercept=0, linetype=":", color="grey")
        plt_obj = (
            plt_obj
            + theme_minimal()
            + labs(title=f"Subgroup {metric} Forest Plot", x=metric, y="Subgroup")
        )
        fname = build_filename_base("subgroup_forest", None, None, "plotnine")
        path_png = os.path.join(output_dir, f"{fname}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt_obj.save(path_png, dpi=DEFAULT_DPI)
    else:
        apply_default_style()
        df = subgroup_results.sort_values("estimate")
        y_pos = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.errorbar(
            df["estimate"],
            y_pos,
            xerr=[df["estimate"] - df["ci_lower"], df["ci_upper"] - df["estimate"]],
            fmt="o",
            color="black",
            ecolor="steelblue",
            elinewidth=2,
            capsize=4,
        )
        if wtp_threshold is not None and metric.lower() == "icer":
            ax.axvline(
                wtp_threshold,
                color="green",
                linestyle="--",
                label=f"WTP ${wtp_threshold:,.0f}",
            )
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["subgroup"])
        ax.set_xlabel(metric)
        ax.set_title(f"Subgroup {metric} Forest Plot")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        save_figure(
            fig,
            output_dir,
            build_filename_base("subgroup_forest", extra=metric.lower()),
        )


def plot_annual_cash_flow(
    bia_results: Dict[str, List[float]],
    output_dir: str = "output/figures/",
    intervention: Optional[str] = None,
    perspective: Optional[str] = None,
):
    """
    Annual cash flow visualization for Budget Impact Analysis.

    Expects keys: 'years', 'investment_costs', 'cost_offsets' (offsets are savings).
    """
    years = bia_results.get("years")
    investment = bia_results.get("investment_costs")
    offsets = bia_results.get("cost_offsets")
    if (
        years is None or investment is None or offsets is None
    ):  # pragma: no cover - guard
        print("Incomplete BIA inputs for annual cash flow plot.")
        return

    net = [i - o for i, o in zip(investment, offsets)]
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(years, investment, color="firebrick", alpha=0.7, label="Investment")
    ax.bar(
        years,
        [-c for c in offsets],
        color="seagreen",
        alpha=0.7,
        label="Offsets (Savings)",
    )
    ax.plot(
        years, net, color="black", marker="o", linewidth=2, label="Net Annual Cash Flow"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Annual Cash Flow ($)")
    ax.set_title("Budget Impact: Annual Cash Flow")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("annual_cash_flow", intervention, perspective),
    )


def plot_efficiency_frontier(
    costs: List[float],
    qalys: List[float],
    strategy_names: List[str],
    output_dir: str = "output/figures/",
):
    """
    Plot non-dominated strategies and the efficiency frontier.
    """
    if not costs or not qalys or not strategy_names:  # pragma: no cover - guard
        print("Insufficient data for efficiency frontier.")
        return

    apply_default_style()
    data = sorted(zip(costs, qalys, strategy_names), key=lambda x: x[0])
    frontier = []
    max_qaly = -np.inf
    for cost, qaly, name in data:
        if qaly > max_qaly:
            frontier.append((cost, qaly, name))
            max_qaly = qaly

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    for cost, qaly, name in data:
        ax.scatter(cost, qaly, label=name)
        ax.annotate(
            name, (cost, qaly), textcoords="offset points", xytext=(5, 5), fontsize=9
        )
    if len(frontier) >= 2:
        fx, fy, _ = zip(*frontier)
        ax.plot(
            fx,
            fy,
            color="black",
            linestyle="-",
            linewidth=2,
            label="Efficiency Frontier",
        )
    ax.set_xlabel("Costs ($)")
    ax.set_ylabel("QALYs")
    ax.set_title("Efficiency Frontier")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("efficiency_frontier"))


def plot_cumulative_nmb(
    nmb_per_cycle: List[float],
    output_dir: str = "output/figures/",
    wtp_label: str = "$/QALY",
):
    """
    Plot cumulative net monetary benefit over time to show time-to-ROI.
    """
    if not nmb_per_cycle:  # pragma: no cover - guard
        print("No NMB per cycle provided for cumulative NMB plot.")
        return

    apply_default_style()
    cumulative = np.cumsum(nmb_per_cycle)
    cycles = np.arange(1, len(cumulative) + 1)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(cycles, cumulative, marker="o", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", label="Break-even")
    ax.fill_between(
        cycles, 0, cumulative, where=(cumulative < 0).tolist(), color="red", alpha=0.1
    )
    ax.fill_between(
        cycles,
        0,
        cumulative,
        where=(cumulative >= 0).tolist(),
        color="green",
        alpha=0.1,
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Cumulative NMB ({wtp_label})")
    ax.set_title("Time-to-ROI: Cumulative Net Monetary Benefit")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("cumulative_nmb"))


def plot_scenario_waterfall(
    base_value: float,
    scenarios: Dict[str, float],
    output_dir: str = "output/figures/",
    metric_label: str = "ICER",
):
    """
    Waterfall chart showing impact of structural/scenario assumptions on a metric.
    """
    if scenarios is None:  # pragma: no cover - guard
        print("No scenarios provided for waterfall plot.")
        return

    apply_default_style()
    labels = ["Base", *list(scenarios.keys()), "Final"]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    current = base_value
    waterfall_vals = [base_value]
    for _name, val in scenarios.items():
        current = val
        waterfall_vals.append(current)
    waterfall_vals.append(current)
    for i in range(1, len(waterfall_vals)):
        start = waterfall_vals[i - 1]
        change = waterfall_vals[i] - start
        color = "seagreen" if change < 0 else "firebrick"
        ax.bar(labels[i], change, bottom=start, color=color, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Scenario Waterfall: {metric_label}")
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("scenario_waterfall"))


def plot_psa_convergence(
    values: List[float], output_dir: str = "output/figures/", metric_label: str = "ICER"
):
    """
    Plot running mean of a PSA metric (ICER or NMB) to show convergence.
    """
    if not values:  # pragma: no cover - guard
        print("No PSA values provided for convergence plot.")
        return

    apply_default_style()
    iterations = np.arange(1, len(values) + 1)
    running_mean = np.cumsum(values) / iterations
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(iterations, running_mean, color="blue", linewidth=2)
    ax.set_xlabel("PSA Iteration")
    ax.set_ylabel(f"Running Mean {metric_label}")
    ax.set_title(f"PSA Convergence: {metric_label}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("psa_convergence", extra=metric_label.lower()),
    )


def plot_model_calibration(
    years: List[int],
    observed_mean: List[float],
    observed_ci: Optional[List[float]],
    predicted_mean: List[float],
    predicted_lower: Optional[List[float]],
    predicted_upper: Optional[List[float]],
    output_dir: str = "output/figures/",
    metric_label: str = "Events",
    use_plotnine: bool = False,
):
    """
    Calibration plot comparing observed vs model-predicted outcomes over time.
    """
    if (
        years is None or observed_mean is None or predicted_mean is None
    ):  # pragma: no cover - guard
        print("Incomplete inputs for calibration plot.")
        return

    if (
        use_plotnine and not PLOTNINE_AVAILABLE
    ):  # pragma: no cover - optional dependency
        print("plotnine not available; falling back to matplotlib.")
        use_plotnine = False

    if use_plotnine:  # pragma: no cover - optional dependency
        df_obs = pd.DataFrame(
            {"year": years, "value": observed_mean, "type": "Observed"}
        )
        df_pred = pd.DataFrame(
            {"year": years, "value": predicted_mean, "type": "Model"}
        )
        df = pd.concat([df_obs, df_pred], ignore_index=True)
        plt_obj = (
            ggplot(df, aes(x="year", y="value", color="type"))
            + geom_line()
            + geom_point()
            + theme_minimal()
            + labs(
                title="Model Calibration: Observed vs Predicted",
                x="Year",
                y=metric_label,
            )
        )
        fname = build_filename_base("model_calibration", None, None, "plotnine")
        os.makedirs(output_dir, exist_ok=True)
        plt_obj.save(os.path.join(output_dir, f"{fname}.png"), dpi=DEFAULT_DPI)
    else:
        apply_default_style()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.errorbar(
            years,
            observed_mean,
            yerr=observed_ci,
            fmt="o",
            color="black",
            label="Observed",
            capsize=4,
        )
        ax.plot(years, predicted_mean, color="blue", marker="s", label="Model")
        if predicted_lower is not None and predicted_upper is not None:
            ax.fill_between(
                years,
                predicted_lower,
                predicted_upper,
                color="blue",
                alpha=0.15,
                label="Model CI",
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(metric_label)
        ax.set_title("Model Calibration: Observed vs Predicted")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("model_calibration"))


def plot_rankogram(
    psa_nmb: pd.DataFrame,
    output_dir: str = "output/figures/",
    use_plotnine: bool = False,
):
    """
    Rank probability plot (rankogram) from PSA NMB draws.

    psa_nmb: DataFrame with columns = strategies, rows = PSA iterations of NMB.
    """
    if psa_nmb is None or psa_nmb.empty:  # pragma: no cover - guard
        print("No PSA NMB data provided for rankogram.")
        return

    ranks = psa_nmb.rank(axis=1, ascending=False, method="first")
    n_strat = psa_nmb.shape[1]
    prob = {
        col: [(ranks[col] == rank).mean() for rank in range(1, n_strat + 1)]
        for col in psa_nmb.columns
    }
    rank_df = pd.DataFrame(prob, index=[f"Rank {i}" for i in range(1, n_strat + 1)])

    if (
        use_plotnine and not PLOTNINE_AVAILABLE
    ):  # pragma: no cover - optional dependency
        print("plotnine not available; falling back to matplotlib.")
        use_plotnine = False

    if use_plotnine:  # pragma: no cover - optional dependency
        long_df = rank_df.T.reset_index().melt(
            id_vars="index", var_name="rank", value_name="prob"
        )
        plt_obj = (
            ggplot(long_df, aes(x="index", y="prob", fill="rank"))
            + geom_bar(stat="identity", position="stack")
            + theme_minimal()
            + labs(
                title="Rank Probability Plot (Rankogram)", x="Strategy", y="Probability"
            )
        )
        fname = build_filename_base("rankogram", None, None, "plotnine")
        os.makedirs(output_dir, exist_ok=True)
        plt_obj.save(os.path.join(output_dir, f"{fname}.png"), dpi=DEFAULT_DPI)
    else:
        apply_default_style()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bottom = np.zeros(n_strat)
        for _i, rank_label in enumerate(rank_df.index):
            values = rank_df.loc[rank_label].values
            ax.bar(psa_nmb.columns, values, bottom=bottom, label=rank_label)
            bottom += values
        ax.set_ylabel("Probability")
        ax.set_title("Rank Probability Plot (Rankogram)")
        ax.legend(title="Rank")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        save_figure(fig, output_dir, build_filename_base("rankogram"))


def plot_price_acceptability_curve(
    wtp_range: List[float],
    break_even_prices: List[float],
    output_dir: str = "output/figures/",
):
    """
    Price acceptability curve showing max justifiable price by WTP.
    """
    if not wtp_range or not break_even_prices:  # pragma: no cover - guard
        print("No data for price acceptability curve.")
        return

    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(wtp_range, break_even_prices, marker="o", linewidth=2)
    ax.set_xlabel("Willingness-to-Pay ($/QALY)")
    ax.set_ylabel("Maximum Justifiable Unit Price ($)")
    ax.set_title("Price Acceptability Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("price_acceptability_curve"))


def plot_resource_constraint(
    annual_resource_use: List[float],
    capacity_limit: float,
    output_dir: str = "output/figures/",
):
    """
    Resource capacity constraint plot showing demand vs capacity.
    """
    if (
        annual_resource_use is None or capacity_limit is None
    ):  # pragma: no cover - guard
        print("Incomplete inputs for resource constraint plot.")
        return

    apply_default_style()
    years = np.arange(1, len(annual_resource_use) + 1)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(
        years,
        annual_resource_use,
        color="steelblue",
        alpha=0.8,
        label="Projected Demand",
    )
    for bar, val in zip(bars, annual_resource_use):
        if val > capacity_limit:
            bar.set_color("firebrick")
    ax.axhline(capacity_limit, color="black", linestyle="--", label="Capacity")
    ax.set_xlabel("Year")
    ax.set_ylabel("Resource Use")
    ax.set_title("Resource Capacity Constraint")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("resource_constraint"))


def plot_expected_loss_curve(
    psa_nmb_by_wtp: Dict[float, pd.DataFrame], output_dir: str = "output/figures/"
):
    """
    Expected opportunity loss vs WTP.

    psa_nmb_by_wtp: mapping of WTP threshold -> DataFrame (rows=iterations, cols=strategies) of NMB.
    """
    if not psa_nmb_by_wtp:  # pragma: no cover - guard
        print("No data for expected loss curve.")
        return

    apply_default_style()
    wtp_values = sorted(psa_nmb_by_wtp.keys())
    losses = []
    for wtp in wtp_values:
        df = psa_nmb_by_wtp[wtp]
        max_nmb = df.max(axis=1)
        # loss if choosing suboptimal is difference between max and each strategy; expected loss is min across strategies of mean loss? standard: expected loss of current best? use max - max mean
        # Here compute expected loss of choosing best-on-average strategy
        mean_nmb = df.mean()
        best_strategy = mean_nmb.idxmax()
        loss = (max_nmb - df[best_strategy]).mean()
        losses.append(loss)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(wtp_values, losses, color="orange", linewidth=2)
    ax.set_xlabel("Willingness-to-Pay ($/QALY)")
    ax.set_ylabel("Expected Opportunity Loss ($)")
    ax.set_title("Expected Loss Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("expected_loss_curve"))


def plot_ly_vs_qaly(
    strategies: List[str],
    ly_gains: List[float],
    utility_gains: List[float],
    output_dir: str = "output/figures/",
):
    """
    Stacked bar chart decomposing QALY gains into life-years and utility components.
    """
    if (
        not strategies or ly_gains is None or utility_gains is None
    ):  # pragma: no cover - guard
        print("Incomplete inputs for LY vs QALY plot.")
        return

    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(strategies, ly_gains, label="Survival Benefit (Life Years)")
    ax.bar(
        strategies, utility_gains, bottom=ly_gains, label="Quality Benefit (Utility)"
    )
    ax.set_ylabel("Total QALYs")
    ax.set_title("Decomposition of Health Benefit")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("ly_vs_qaly"))


def plot_decision_reversal_matrix(
    hs_nmb: List[float],
    soc_nmb: List[float],
    strategy_names: List[str],
    output_dir: str = "output/figures/",
):
    """
    Concordance plot showing decision reversals between health system and societal perspectives.
    """
    if (
        hs_nmb is None or soc_nmb is None or strategy_names is None
    ):  # pragma: no cover - guard
        print("Incomplete inputs for decision reversal matrix.")
        return

    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.fill_between(
        [0, max(hs_nmb + soc_nmb)],
        0,
        max(hs_nmb + soc_nmb),
        color="lightgreen",
        alpha=0.1,
        label="Win-Win",
    )
    ax.fill_between(
        [min(hs_nmb + soc_nmb), 0],
        0,
        max(soc_nmb),
        color="gold",
        alpha=0.1,
        label="HS Negative / Soc Positive",
    )
    ax.fill_between(
        [0, max(hs_nmb + soc_nmb)],
        min(soc_nmb),
        0,
        color="salmon",
        alpha=0.1,
        label="HS Positive / Soc Negative",
    )
    for x, y, name in zip(hs_nmb, soc_nmb, strategy_names):
        ax.scatter(x, y, s=80)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Health System NMB")
    ax.set_ylabel("Societal NMB")
    ax.set_title("Decision Concordance by Perspective")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("decision_reversal_matrix"))


def plot_societal_drivers(
    intervention_name: str,
    drivers: Dict[str, float],
    output_dir: str = "output/figures/",
):
    """
    Decomposition of societal perspective components (e.g., productivity, carer time).
    """
    if not drivers:  # pragma: no cover - guard
        print("No drivers provided for societal decomposition plot.")
        return

    apply_default_style()
    categories = list(drivers.keys())
    values = list(drivers.values())
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(categories, values, color="slateblue", alpha=0.8)
    ax.set_ylabel("Cost Contribution ($)")
    ax.set_title(f"Societal Drivers: {intervention_name}")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_figure(
        fig, output_dir, build_filename_base("societal_drivers", intervention_name)
    )


def plot_evp_curve(
    wtp_values: List[float],
    hs_nmb: Dict[str, List[float]],
    soc_nmb: Dict[str, List[float]],
    output_dir: str = "output/figures/",
):
    """
    Expected Value of Perspective curve.

    hs_nmb / soc_nmb: mapping strategy -> list of expected NMB at each WTP (aligned to wtp_values).
    """
    if not wtp_values or not hs_nmb or not soc_nmb:  # pragma: no cover - guard
        print("Incomplete inputs for EVP curve.")
        return

    apply_default_style()
    losses = []
    for i, _wtp in enumerate(wtp_values):
        hs_best = max(hs_nmb.items(), key=lambda kv: kv[1][i])[0]
        soc_best_nmb = max(soc_nmb.items(), key=lambda kv: kv[1][i])[1][i]
        hs_choice_nmb = soc_nmb.get(hs_best, [0] * len(wtp_values))[i]
        losses.append(soc_best_nmb - hs_choice_nmb)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(wtp_values, losses, color="purple", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Willingness-to-Pay ($/QALY)")
    ax.set_ylabel("Expected Loss from Perspective Choice ($)")
    ax.set_title("Expected Value of Perspective (EVP) Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("evp_curve"))


def plot_structural_tornado(
    base_nmb: float,
    scenario_ranges: Dict[str, tuple],
    output_dir: str = "output/figures/",
):
    """
    Tornado plot including structural parameters (e.g., perspective).

    scenario_ranges: dict of parameter -> (low_nmb, high_nmb)
    """
    if not scenario_ranges:  # pragma: no cover - guard
        print("No scenario ranges provided for structural tornado.")
        return

    apply_default_style()
    params = list(scenario_ranges.keys())
    diffs = []
    lows = []
    highs = []
    for low, high in scenario_ranges.values():
        diffs.append(abs(high - low))
        lows.append(low)
        highs.append(high)
    order = np.argsort(diffs)[::-1]
    params = [params[i] for i in order]
    lows = [lows[i] for i in order]
    highs = [highs[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    y_pos = np.arange(len(params))
    for i, (low, high) in enumerate(zip(lows, highs)):
        ax.plot([low, high], [y_pos[i]] * 2, color="steelblue", linewidth=6)
        ax.scatter([low, high], [y_pos[i]] * 2, color="black")
    ax.axvline(base_nmb, color="red", linestyle="--", label="Base NMB")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel("Net Monetary Benefit")
    ax.set_title("Structural Sensitivity Tornado (including Perspective)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("structural_tornado"))


# ---------------------------------------------------------------------------
# Dashboard composer
# ---------------------------------------------------------------------------


def compose_dashboard(
    image_paths: List[str],
    output_dir: str = "output/figures/",
    filename_base: str = "dashboard",
    ncols: int = 2,
):
    """
    Simple dashboard grid that stitches existing plot images together.
    """
    if not image_paths:  # pragma: no cover - guard
        print("No images provided for dashboard composition.")
        return
    import matplotlib.image as mpimg

    existing_paths = [p for p in image_paths if os.path.exists(p)]
    if not existing_paths:  # pragma: no cover - guard
        print("No existing images found for dashboard composition.")
        return
    nrows = int(np.ceil(len(existing_paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, path in zip(axes, existing_paths):
        try:
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(path), fontsize=9)
        except FileNotFoundError:  # pragma: no cover - guard
            ax.text(
                0.5, 0.5, f"Missing: {os.path.basename(path)}", ha="center", va="center"
            )
            ax.axis("off")
    for ax in axes[len(existing_paths) :]:
        ax.axis("off")
    plt.tight_layout()
    save_figure(fig, output_dir, filename_base)


def compose_bia_dashboard(output_dir: str = "output/figures/"):
    """Compose Budget Impact Analysis dashboard."""
    from glob import glob

    images = []
    images.extend(sorted(glob(os.path.join(output_dir, "annual_cash_flow*.png"))))
    images.extend(
        sorted(glob(os.path.join(output_dir, "net_cash_flow_waterfall*.png")))
    )
    images.extend(sorted(glob(os.path.join(output_dir, "affordability_ribbon*.png"))))
    compose_dashboard(images, output_dir=output_dir, filename_base="dashboard_bia")


def compose_equity_dashboard(
    intervention_names: List[str], output_dir: str = "output/figures/"
):
    """Compose Equity/DCEA dashboard across interventions."""
    images = []
    for name in intervention_names:
        base = slugify(name)
        images.append(os.path.join(output_dir, f"equity_impact_plane_{base}.png"))
        images.append(os.path.join(output_dir, f"lorenz_curve_{base}.png"))
        images.append(os.path.join(output_dir, f"equity_efficiency_plane_{base}.png"))
    images = [img for img in images if os.path.exists(img)]
    if not images:  # pragma: no cover - guard
        print("No equity images found for dashboard.")
        return
    compose_dashboard(images, output_dir=output_dir, filename_base="dashboard_equity")


# ---------------------------------------------------------------------------
# Additional gap-closing plots
# ---------------------------------------------------------------------------


def plot_survival_gof(
    km_time: List[float],
    km_survival: List[float],
    fitted_models: Dict[str, Dict[str, List[float]]],
    output_dir: str = "output/figures/",
):
    """
    Overlay Kaplan-Meier survival with parametric fits for GOF assessment.
    fitted_models: {name: {"time": [...], "survival": [...], "aic": optional}}
    """
    if not km_time or not km_survival:  # pragma: no cover - guard
        print("KM data missing for survival GOF plot.")
        return
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.step(km_time, km_survival, where="post", label="KM", color="black")
    for name, model in fitted_models.items():
        t = model.get("time", [])
        s = model.get("survival", [])
        if not t or not s:
            continue
        aic = model.get("aic")
        label = f"{name}" + (f" (AIC {aic:.1f})" if aic is not None else "")
        ax.plot(t, s, linestyle="--", label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title("Survival Goodness-of-Fit")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("survival_gof"))


def plot_equity_efficiency_plane(
    strategies: List[str],
    efficiency_gains: List[float],
    equity_gains: List[float],
    output_dir: str = "output/figures/",
):
    """Equity-Efficiency impact plane."""
    if (
        not strategies or efficiency_gains is None or equity_gains is None
    ):  # pragma: no cover - guard
        print("Inputs missing for equity-efficiency plane.")
        return
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.scatter(equity_gains, efficiency_gains)
    for x, y, name in zip(equity_gains, efficiency_gains, strategies):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Equity Impact (reduction in inequality)")
    ax.set_ylabel("Efficiency Impact (Net Health Benefit)")
    ax.set_title("Equity-Efficiency Impact Plane")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("equity_efficiency_plane"))


def plot_inequality_staircase(
    stages: List[str], relative_risks: List[float], output_dir: str = "output/figures/"
):
    """Staircase of inequality across pathway stages."""
    if not stages or not relative_risks:  # pragma: no cover - guard
        print("Inputs missing for inequality staircase.")
        return
    apply_default_style()
    cumulative = np.cumprod(relative_risks)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.step(stages, cumulative, where="mid", marker="o")
    ax.axhline(1, linestyle="--", color="grey", label="Perfect Equality")
    ax.set_ylabel("Cumulative Inequality Ratio (Advantaged/Disadvantaged)")
    ax.set_title("Staircase of Inequality")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("inequality_staircase"))


def plot_financial_risk_protection(
    strategies: List[str],
    poverty_cases_averted: List[float],
    output_dir: str = "output/figures/",
):
    """Financial risk protection (ECEA) plot."""
    if not strategies or poverty_cases_averted is None:  # pragma: no cover - guard
        print("Inputs missing for financial risk protection plot.")
        return
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(strategies, poverty_cases_averted, color="purple", alpha=0.7)
    ax.set_ylabel("Cases of Catastrophic Expenditure Averted")
    ax.set_title("Financial Risk Protection")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, output_dir, build_filename_base("financial_risk_protection"))


def plot_affordability_ribbon(
    years: List[int],
    mean: List[float],
    lower: List[float],
    upper: List[float],
    output_dir: str = "output/figures/",
    intervention: Optional[str] = None,
    perspective: Optional[str] = None,
):
    """Affordability ribbon for BIA sensitivity."""
    if (
        years is None or mean is None or lower is None or upper is None
    ):  # pragma: no cover - guard
        print("Inputs missing for affordability ribbon.")
        return
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(years, mean, color="black", label="Mean Budget Impact")
    ax.fill_between(
        years, lower, upper, color="skyblue", alpha=0.4, label="Sensitivity Range"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Budget Impact ($)")
    ax.set_title("Affordability Ribbon")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("affordability_ribbon", intervention, perspective),
    )


def plot_threshold_crossing(
    param_range: List[float],
    nmb_values: List[float],
    param_name: str,
    output_dir: str = "output/figures/",
):
    """Threshold crossing plot showing where decision flips."""
    if param_range is None or nmb_values is None:  # pragma: no cover - guard
        print("Inputs missing for threshold crossing plot.")
        return
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(param_range, nmb_values, linewidth=2.5, color="blue")
    crossings = np.where(np.diff(np.sign(nmb_values)))[0]
    if len(crossings) > 0:
        crossing_x = param_range[crossings[0]]
        ax.axvline(
            crossing_x,
            color="red",
            linestyle="--",
            label=f"Switching Point: {crossing_x:.2f}",
        )
        ax.scatter([crossing_x], [0], color="red", s=80, zorder=5)
    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(
        param_range,
        0,
        nmb_values,
        where=(np.array(nmb_values) > 0).tolist(),
        color="green",
        alpha=0.1,
        label="Cost-Effective",
    )
    ax.fill_between(
        param_range,
        0,
        nmb_values,
        where=(np.array(nmb_values) < 0).tolist(),
        color="red",
        alpha=0.1,
        label="Not Cost-Effective",
    )
    ax.set_xlabel(param_name)
    ax.set_ylabel("Net Monetary Benefit ($)")
    ax.set_title(f"Threshold Analysis: {param_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(
        fig, output_dir, build_filename_base("threshold_crossing", extra=param_name)
    )


def plot_structural_waterfall(
    base_value: float,
    scenarios: Dict[str, float],
    output_dir: str = "output/figures/",
    metric_label: str = "ICER",
):
    """Structural scenario waterfall (e.g., discount rate, horizon, perspective)."""
    plot_scenario_waterfall(base_value, scenarios, output_dir, metric_label)


def plot_net_cash_flow_waterfall(
    investment_costs: List[float],
    cost_offsets: List[float],
    years: List[int],
    output_dir: str = "output/figures/",
    intervention: Optional[str] = None,
    perspective: Optional[str] = None,
):
    """Waterfall showing yearly net cash flows."""
    if (
        years is None or investment_costs is None or cost_offsets is None
    ):  # pragma: no cover - guard
        print("Inputs missing for net cash flow waterfall.")
        return
    apply_default_style()
    net: List[float] = [
        float(i) - float(o) for i, o in zip(investment_costs, cost_offsets)
    ]
    labels = [f"Year {y}" for y in years]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    running: float = 0.0
    for lbl, val in zip(labels, net):
        color = "seagreen" if val < 0 else "firebrick"
        running_val = float(val)
        ax.bar(lbl, running_val, bottom=running, color=color, alpha=0.8)
        running += running_val
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net Cash Flow ($)")
    ax.set_title("Net Cash Flow Waterfall")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("net_cash_flow_waterfall", intervention, perspective),
    )


def plot_annual_cash_flow(
    years: List[int],
    gross_costs: List[float],
    net_costs: List[float],
    output_dir: str = "output/figures/",
    intervention: Optional[str] = None,
    perspective: Optional[str] = None,
):
    """
    Plot annual cash flows (Gross vs Net) for Budget Impact Analysis.
    """
    if years is None or gross_costs is None or net_costs is None:
        print("Inputs missing for annual cash flow plot.")
        return

    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    width = 0.35
    x = np.arange(len(years))

    ax.bar(
        x - width / 2,
        gross_costs,
        width,
        label="Gross Cost",
        color="firebrick",
        alpha=0.8,
    )
    ax.bar(x + width / 2, net_costs, width, label="Net Cost", color="navy", alpha=0.8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cost ($)")
    ax.set_title(f"Annual Budget Impact: {intervention}")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("annual_cash_flow", intervention, perspective),
    )


def plot_discordance_loss(
    discordance_results: List[Dict],
    output_dir: str = "output/figures/",
):
    """
    Plot the Opportunity Cost (Loss) from Discordance across interventions.
    """
    if not discordance_results:
        print("No discordance results to plot.")
        return

    apply_default_style()

    names = [r["intervention"] for r in discordance_results]
    losses = [r["loss_from_discordance"] for r in discordance_results]

    # Filter out zero losses if desired, or keep to show alignment
    # For impact, we highlight the non-zero ones

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    bars = ax.bar(names, losses, color="darkred", alpha=0.7)

    ax.set_ylabel("Opportunity Cost ($)")
    ax.set_title("Value of Perspective: Cost of Decision Discordance")
    ax.grid(alpha=0.3, axis="y")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${height:,.0f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        build_filename_base("discordance_loss"),
    )


def plot_inequality_aversion_sensitivity(
    sensitivity_results: pd.DataFrame,
    intervention_name: str,
    output_dir: str = "output/figures/",
):
    """
    Plots the results of the inequality aversion sensitivity analysis.

    Args:
        sensitivity_results: DataFrame with epsilon, atkinson_index, and ede_net_benefit.
        intervention_name: Name of the intervention.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    apply_default_style()

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    color = "tab:red"
    ax1.set_xlabel("Inequality Aversion Parameter (ε)")
    ax1.set_ylabel("Atkinson Index (Inequality)", color=color)
    ax1.plot(
        sensitivity_results["epsilon"],
        sensitivity_results["atkinson_index"],
        color=color,
        marker="o",
        label="Atkinson Index",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Equally Distributed Equivalent (EDE) Net Benefit ($)", color=color)
    ax2.plot(
        sensitivity_results["epsilon"],
        sensitivity_results["ede_net_benefit"],
        color=color,
        marker="s",
        linestyle="--",
        label="Social Welfare (EDE)",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Inequality Aversion Sensitivity: {intervention_name}")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    save_figure(
        fig,
        output_dir,
        build_filename_base("inequality_sensitivity", intervention_name),
    )


def plot_comparative_ce_plane_with_delta(all_results, output_dir="output/figures/"):
    """
    Create comparative Cost-Effectiveness Plane with Delta subplot.
    3 Subplots: Health System, Societal, Delta (Societal - HS).
    """
    apply_default_style()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(
        "Comparative Cost-Effectiveness Plane with Perspective Delta\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )

    perspectives = ["Health System", "Societal", "Delta (Societal - HS)"]
    
    # Define colors for interventions
    model_names = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = dict(zip(model_names, colors))

    # Collect Delta Data for Violin Plot
    delta_costs_data = []
    delta_costs_labels = []
    
    for model_name in model_names:
        psa_results = all_results[model_name]
        c = color_map[model_name]
        
        # Extract data
        # Health System
        inc_cost_hs = psa_results["inc_cost_hs"]
        inc_qaly_hs = psa_results["inc_qaly_hs"]
        
        # Societal
        inc_cost_soc = psa_results["inc_cost_soc"]
        inc_qaly_soc = psa_results["inc_qaly_soc"]
        
        # Delta
        delta_cost = inc_cost_soc - inc_cost_hs
        delta_qaly = inc_qaly_soc - inc_qaly_hs
        
        # Store for Violin
        delta_costs_data.append(delta_cost)
        delta_costs_labels.append(model_name)
        
        # Plot Health System
        axes[0].scatter(
            inc_qaly_hs,
            inc_cost_hs,
            label=model_name,
            color=c,
            alpha=0.15,
            s=10,
            edgecolors="none"
        )
        # Plot Mean Point
        axes[0].scatter(
            np.mean(inc_qaly_hs),
            np.mean(inc_cost_hs),
            color=c,
            s=100,
            edgecolors="black",
            marker="o"
        )

        # Plot Societal
        axes[1].scatter(
            inc_qaly_soc,
            inc_cost_soc,
            label=model_name,
            color=c,
            alpha=0.15,
            s=10,
            edgecolors="none"
        )
        axes[1].scatter(
            np.mean(inc_qaly_soc),
            np.mean(inc_cost_soc),
            color=c,
            s=100,
            edgecolors="black",
            marker="o"
        )
        
    # Plot Delta (Violin)
    # axes[2] is now a Violin Plot of Delta Costs
    parts = axes[2].violinplot(
        delta_costs_data,
        showmeans=True,
        showextrema=False,
        showmedians=False
    )
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    # Fix means color
    parts['cmeans'].set_color('black')

    # Formatting
    for i, ax in enumerate(axes):
        ax.set_title(perspectives[i], fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        if i < 2:
            ax.set_xlabel("Incremental QALYs", fontsize=12)
            ax.set_ylabel("Incremental Cost ($)", fontsize=12)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
            ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
            
            # WTP Threshold line (50k)
            xlim = ax.get_xlim()
            x_vals = np.array(xlim)
            y_vals = 50000 * x_vals
            ax.plot(x_vals, y_vals, 'k--', alpha=0.5, label="WTP $50k/QALY")
            ax.set_xlim(xlim) # Restore limits
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:,.0f}k"))
        else:
            # Delta Plot Formatting
            ax.set_ylabel("Delta Cost (Societal - HS)", fontsize=12)
            ax.set_xticks(np.arange(1, len(model_names) + 1))
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:,.0f}k"))

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Filter for unique labels (interventions + WTP)
    by_label = dict(zip(labels, handles))
    # We only want intervention labels, WTP is common
    
    # Create custom legend for interventions
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color_map[name], markersize=10) for name in model_names]
    legend_elements.append(Line2D([0], [0], color='k', linestyle='--', label='WTP $50k/QALY'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(model_names)+1, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(
        fig,
        output_dir,
        "cost_effectiveness_plane_comparative_with_delta",
    )


def plot_comparative_ceac_with_delta(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create comparative CEAC with Delta subplot.
    3 Subplots: Health System, Societal, Delta (Societal - HS).
    """
    apply_default_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(
        "Comparative Cost-Effectiveness Acceptability Curves with Perspective Delta\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )
    
    perspectives = ["Health System", "Societal", "Delta (Societal - HS)"]
    model_names = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = dict(zip(model_names, colors))
    
    for model_name in model_names:
        psa_results = all_results[model_name]
        c = color_map[model_name]
        
        # Calculate CEAC for Health System
        # We need to manually calculate probabilities because ProbabilisticSensitivityAnalysis.calculate_ceac 
        # uses specific columns. We can reuse the logic.
        
        prob_ce_hs = []
        prob_ce_soc = []
        
        for wtp in wtp_thresholds:
            # HS
            nmb_hs = (psa_results["inc_qaly_hs"] * wtp) - psa_results["inc_cost_hs"]
            prob_hs = np.mean(nmb_hs > 0)
            prob_ce_hs.append(prob_hs)
            
            # Societal
            nmb_soc = (psa_results["inc_qaly_soc"] * wtp) - psa_results["inc_cost_soc"]
            prob_soc = np.mean(nmb_soc > 0)
            prob_ce_soc.append(prob_soc)
            
        prob_ce_hs = np.array(prob_ce_hs)
        prob_ce_soc = np.array(prob_ce_soc)
        delta_prob = prob_ce_soc - prob_ce_hs
        
        # Plot HS
        axes[0].plot(wtp_thresholds, prob_ce_hs, label=model_name, color=c, linewidth=2)
        
        # Plot Societal
        axes[1].plot(wtp_thresholds, prob_ce_soc, label=model_name, color=c, linewidth=2)
        
        # Plot Delta
        axes[2].plot(wtp_thresholds, delta_prob, label=model_name, color=c, linewidth=2)
        
    # Formatting
    for i, ax in enumerate(axes):
        ax.set_title(perspectives[i], fontsize=14, fontweight="bold")
        ax.set_xlabel("WTP Threshold ($/QALY)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}k"))
        
        if i < 2:
            ax.set_ylabel("Probability Cost-Effective", fontsize=12)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel("Delta Probability (Soc - HS)", fontsize=12)
            # Delta can be negative or positive, generally between -1 and 1
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color_map[name], lw=2, label=name) for name in model_names]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(model_names), frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(
        fig,
        output_dir,
        "ceac_comparative_with_delta",
    )


def plot_comparative_evpi_with_delta(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create comparative EVPI plot with Delta subplot.
    3 Subplots: Health System, Societal, Delta (Societal - HS).
    """
    apply_default_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(
        "Comparative Expected Value of Perfect Information (EVPI) with Perspective Delta\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )
    
    perspectives = ["Health System", "Societal", "Delta (Societal - HS)"]
    model_names = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = dict(zip(model_names, colors))
    
    for model_name in model_names:
        psa_results = all_results[model_name]
        c = color_map[model_name]
        
        # Calculate EVPI for both perspectives
        # We need to temporarily modify calculate_evpi or manually calculate
        # calculate_evpi takes psa_results and uses "qaly_sc", "cost_sc" etc.
        # We need to map columns.
        
        evpi_hs = []
        evpi_soc = []
        
        # Helper to calculate EVPI given columns
        def calc_evpi_vals(df, col_c_sc, col_q_sc, col_c_nt, col_q_nt, wtp_list):
            vals = []
            for wtp in wtp_list:
                nmb_sc = (df[col_q_sc] * wtp) - df[col_c_sc]
                nmb_nt = (df[col_q_nt] * wtp) - df[col_c_nt]
                nmb_matrix = np.column_stack([nmb_sc, nmb_nt])
                max_nmb_per_sim = np.max(nmb_matrix, axis=1)
                expected_nmb_with_perfect_info = np.mean(max_nmb_per_sim)
                current_optimal_nmb = max(np.mean(nmb_sc), np.mean(nmb_nt))
                evpi = max(0.0, expected_nmb_with_perfect_info - current_optimal_nmb)
                vals.append(evpi)
            return np.array(vals)

        # Health System Columns
        evpi_hs = calc_evpi_vals(
            psa_results, 
            "cost_sc_hs", "qaly_sc_hs", "cost_nt_hs", "qaly_nt_hs", 
            wtp_thresholds
        )
        
        # Societal Columns
        evpi_soc = calc_evpi_vals(
            psa_results, 
            "cost_sc_soc", "qaly_sc_soc", "cost_nt_soc", "qaly_nt_soc", 
            wtp_thresholds
        )
        
        delta_evpi = evpi_soc - evpi_hs
        
        # Plot HS
        axes[0].plot(wtp_thresholds, evpi_hs, label=model_name, color=c, linewidth=2)
        
        # Plot Societal
        axes[1].plot(wtp_thresholds, evpi_soc, label=model_name, color=c, linewidth=2)
        
        # Plot Delta
        axes[2].plot(wtp_thresholds, delta_evpi, label=model_name, color=c, linewidth=2)
        
    # Formatting
    for i, ax in enumerate(axes):
        ax.set_title(perspectives[i], fontsize=14, fontweight="bold")
        ax.set_xlabel("WTP Threshold ($/QALY)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}k"))
        
        if i < 2:
            ax.set_ylabel("EVPI ($)", fontsize=12)
        else:
            ax.set_ylabel("Delta EVPI (Soc - HS)", fontsize=12)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color_map[name], lw=2, label=name) for name in model_names]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(model_names), frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(
        fig,
        output_dir,
        "expected_value_perfect_information_comparative_with_delta",
    )


def plot_comparative_evppi_with_delta(voi_results, output_dir="output/figures/"):
    """
    Create comparative EVPPI plot with Delta subplot.
    3 Subplots: Health System, Societal, Delta (Societal - HS).
    """
    apply_default_style()
    
    if not voi_results:
        return

    # Collect data
    data_rows = []
    
    for model_name, report in voi_results.items():
        evppi_data = report["value_of_information"]["evppi_by_parameter_group"]
        wtp_thresholds = report["value_of_information"]["wtp_thresholds"]
        
        for key, values in evppi_data.items():
            if key.endswith("_HS"):
                perspective = "Health System"
                group_name = key.replace("EVPPI_", "").replace("_HS", "").replace("_", " ")
            elif key.endswith("_Soc"):
                perspective = "Societal"
                group_name = key.replace("EVPPI_", "").replace("_Soc", "").replace("_", " ")
            else:
                continue # Skip generic

            if "Cost Parameters" not in group_name and "QALY Parameters" not in group_name:
                continue

            for i, wtp in enumerate(wtp_thresholds):
                data_rows.append({
                    "Intervention": model_name,
                    "Perspective": perspective,
                    "Parameter Group": group_name,
                    "WTP": wtp,
                    "EVPPI": values[i],
                    "Index": i # Keep index to match for delta
                })
                
    if not data_rows:
        return
        
    df = pd.DataFrame(data_rows)
    
    # Calculate Delta
    # Pivot to get HS and Soc side by side
    df_pivot = df.pivot_table(
        index=["Intervention", "Parameter Group", "WTP", "Index"], 
        columns="Perspective", 
        values="EVPPI"
    ).reset_index()
    
    df_pivot["Delta"] = df_pivot["Societal"] - df_pivot["Health System"]
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300, sharey=False)
    fig.suptitle(
        "Comparative Expected Value of Partial Perfect Information (EVPPI) with Perspective Delta\n(2024 NZD equivalent)",
        fontsize=16,
        fontweight="bold",
    )
    
    perspectives = ["Health System", "Societal", "Delta (Societal - HS)"]
    
    linestyles = {"Cost Parameters": "-", "QALY Parameters": "--"}
    interventions = df["Intervention"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(interventions)))
    color_map = dict(zip(interventions, colors))
    
    # Plot HS and Soc
    for i, perspective in enumerate(["Health System", "Societal"]):
        ax = axes[i]
        subset = df[df["Perspective"] == perspective]
        
        for intervention in interventions:
            int_data = subset[subset["Intervention"] == intervention]
            c = color_map[intervention]
            for group in ["Cost Parameters", "QALY Parameters"]:
                group_data = int_data[int_data["Parameter Group"] == group]
                if not group_data.empty:
                    ax.plot(group_data["WTP"], group_data["EVPPI"], color=c, linestyle=linestyles[group], linewidth=2)
                    
    # Plot Delta
    ax = axes[2]
    for intervention in interventions:
        int_data = df_pivot[df_pivot["Intervention"] == intervention]
        c = color_map[intervention]
        for group in ["Cost Parameters", "QALY Parameters"]:
            group_data = int_data[int_data["Parameter Group"] == group]
            if not group_data.empty:
                ax.plot(group_data["WTP"], group_data["Delta"], color=c, linestyle=linestyles[group], linewidth=2)
                
    # Formatting
    for i, ax in enumerate(axes):
        ax.set_title(perspectives[i], fontsize=14, fontweight="bold")
        ax.set_xlabel("WTP Threshold ($/QALY)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}k"))
        
        if i < 2:
            ax.set_ylabel("EVPPI ($)", fontsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        else:
            ax.set_ylabel("Delta EVPPI ($)", fontsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Legend
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    legend_elements = []
    legend_elements.append(mpatches.Patch(color='none', label=r'$\bf{Interventions:}$'))
    for intervention in interventions:
        legend_elements.append(Line2D([0], [0], color=color_map[intervention], lw=2, label=intervention))
    
    legend_elements.append(mpatches.Patch(color='none', label=' '))
    legend_elements.append(mpatches.Patch(color='none', label=r'$\bf{Parameters:}$'))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', label='Cost Parameters'))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label='QALY Parameters'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(
        fig,
        output_dir,
        "expected_value_partial_perfect_information_comparative_with_delta",
    )


def plot_comparative_bia_line(
    bia_results: Dict[str, pd.DataFrame],
    output_dir: str = "output/figures/",
):
    """
    Plot comparative Budget Impact Analysis (Gross, Net, and Offsets) for all interventions.
    """
    apply_default_style()
    # 1 row, 3 columns, side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, sharex=True)
    
    # Define styles
    styles = {
        "HPV Vaccination": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "Smoking Cessation": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
        "Hepatitis C Therapy": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
        "Childhood Obesity Prevention": {"color": "#d62728", "linestyle": ":", "marker": "D"},
        "Housing Insulation": {"color": "#9467bd", "linestyle": "-", "marker": "v"},
    }
    
    import matplotlib.ticker as mtick

    # Plot Gross Cost (Left Panel)
    ax_gross = axes[0]
    for name, df in bia_results.items():
        years = df["year"].tolist()
        gross_costs = df["gross_cost"].tolist()
        style = styles.get(name, {"linestyle": "-", "marker": "o"})
        ax_gross.plot(years, gross_costs, label=name, linewidth=2.5, alpha=0.8, **style)
    
    ax_gross.set_ylabel("Gross Budget Impact (NZ$)", fontsize=12)
    ax_gross.set_xlabel("Year", fontsize=12)
    ax_gross.set_title("A. Gross Budget Impact\n(Total Investment)", fontsize=14, fontweight="bold", loc="left")
    ax_gross.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax_gross.grid(True, alpha=0.3)
    # Legend only on the first plot to avoid clutter, or outside
    ax_gross.legend(fontsize=9, loc="upper left")

    # Plot Net Cost (Center Panel)
    ax_net = axes[1]
    for name, df in bia_results.items():
        years = df["year"].tolist()
        net_costs = df["net_cost"].tolist()
        style = styles.get(name, {"linestyle": "-", "marker": "o"})
        ax_net.plot(years, net_costs, label=name, linewidth=2.5, alpha=0.8, **style)
        
    ax_net.axhline(0, color="black", linestyle="-", linewidth=1)
    ax_net.set_xlabel("Year", fontsize=12)
    # ax_net.set_ylabel("Net Budget Impact (NZ$)", fontsize=12) # Share y-axis label? No, scales might differ
    ax_net.set_title("B. Net Budget Impact\n(Investment - Offsets)", fontsize=14, fontweight="bold", loc="left")
    ax_net.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax_net.grid(True, alpha=0.3)

    # Plot Offsets (Right Panel)
    ax_offsets = axes[2]
    for name, df in bia_results.items():
        years = df["year"].tolist()
        # Calculate offsets if not explicitly in df, but usually it is. 
        # If not, Gross - Net = Offsets. 
        # Checking keys: 'offsets' is in the JSON structure we saw earlier.
        if "offsets" in df.columns:
            offsets = df["offsets"].tolist()
        else:
            offsets = (df["gross_cost"] - df["net_cost"]).tolist()
            
        style = styles.get(name, {"linestyle": "-", "marker": "o"})
        ax_offsets.plot(years, offsets, label=name, linewidth=2.5, alpha=0.8, **style)

    ax_offsets.set_xlabel("Year", fontsize=12)
    ax_offsets.set_title("C. Cost Offsets\n(Avoided Standard Care)", fontsize=14, fontweight="bold", loc="left")
    ax_offsets.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax_offsets.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save as dashboard_bia_v2.png
    save_figure(fig, output_dir, "dashboard_bia_v2")
