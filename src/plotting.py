import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import os
from typing import Dict
from .value_of_information import ProbabilisticSensitivityAnalysis, calculate_evpi
from graphviz import Digraph

def plot_decision_tree(model_name: str, params: Dict, output_dir: str = "output/figures/"):
    """
    Generates a decision tree diagram for a given intervention.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dot = Digraph(comment=f'Decision Tree for {model_name}')

    # Nodes
    dot.node('D', 'Decision', shape='box')
    dot.node('SC', 'Standard Care')
    dot.node('NT', 'New Treatment')

    # Edges from Decision
    dot.edge('D', 'SC')
    dot.edge('D', 'NT')

    # States for Standard Care
    for i, state in enumerate(params['states']):
        dot.node(f'SC_{state}', f'{state}\nCost: {params["costs"]["health_system"]["standard_care"] [i]}\nQALY: {params["qalys"]["standard_care"] [i]}')
        dot.edge('SC', f'SC_{state}')

    # States for New Treatment
    for i, state in enumerate(params['states']):
        dot.node(f'NT_{state}', f'{state}\nCost: {params["costs"]["health_system"]["new_treatment"] [i]}\nQALY: {params["qalys"]["new_treatment"] [i]}')
        dot.edge('NT', f'NT_{state}')
        
    # Transition probabilities
    for i, from_state in enumerate(params['states']):
        for j, to_state in enumerate(params['states']):
            prob_sc = params['transition_matrices']['standard_care'][i][j]
            if prob_sc > 0:
                dot.edge(f'SC_{from_state}', f'SC_{to_state}', label=str(prob_sc))
            
            prob_nt = params['transition_matrices']['new_treatment'][i][j]
            if prob_nt > 0:
                dot.edge(f'NT_{from_state}', f'NT_{to_state}', label=str(prob_nt))

    # Render and save the diagram
    dot.render(f"{output_dir}/decision_tree_{model_name.lower().replace(' ', '_')}", view=False, format='png')
    dot.render(f"{output_dir}/decision_tree_{model_name.lower().replace(' ', '_')}", view=False, format='pdf')

def plot_cost_effectiveness_plane(all_results, output_dir="output/figures/"):
    """
    Create separate cost-effectiveness plane plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Plane: Comparison Across Interventions\n(WTP = $50,000/QALY, 2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    for model_name in model_names:
        data = all_results[model_name]
        ax.scatter(data['inc_cost'], data['inc_qaly'], alpha=0.5, s=10, label=model_name)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Incremental Costs ($)", fontsize=12)
    ax.set_ylabel("Incremental QALYs", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cost_effectiveness_plane.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/cost_effectiveness_plane.pdf", bbox_inches="tight")
    plt.close()


def plot_ceac(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create separate cost-effectiveness acceptability curve plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Acceptability Curve: Comparison Across Interventions\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    # Create a dummy PSA object to access calculate_ceac method
    psa_calculator = ProbabilisticSensitivityAnalysis(model_func=None, parameters=None, wtp_threshold=50000)

    model_names = list(all_results.keys())
    for model_name in model_names:
        psa_results = all_results[model_name] # This is the DataFrame from psa.run_psa()
        
        # Calculate CEAC for the societal perspective (as psa_run_cea_wrapper is currently set to societal)
        ceac_df = psa_calculator.calculate_ceac(psa_results, wtp_values=wtp_thresholds)
        
        ax.plot(
            ceac_df['wtp_threshold'],
            ceac_df['probability_cost_effective'],
            label=f"{model_name} - Societal",
            linewidth=2,
        )

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("Probability of Cost-Effectiveness", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/cost_effectiveness_acceptability_curve.png", bbox_inches="tight"
    )
    plt.savefig(
        f"{output_dir}/cost_effectiveness_acceptability_curve.pdf", bbox_inches="tight"
    )
    plt.close()


def plot_ceaf(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create separate cost-effectiveness acceptability frontier plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Cost-Effectiveness Acceptability Frontier: Comparison Across Interventions\n(WTP = $50,000/QALY, 2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    # Create a dummy PSA object to access calculate_ceac method
    psa_calculator = ProbabilisticSensitivityAnalysis(model_func=None, parameters=None, wtp_threshold=50000)

    model_names = list(all_results.keys())
    ceac_data_list = []
    for model_name in model_names:
        psa_results = all_results[model_name]
        ceac_df = psa_calculator.calculate_ceac(psa_results, wtp_values=wtp_thresholds)
        ceac_data_list.append(ceac_df['probability_cost_effective'])
        
        ax.plot(
            ceac_df['wtp_threshold'],
            ceac_df['probability_cost_effective'],
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
    plt.savefig(
        f"{output_dir}/cost_effectiveness_acceptability_frontier.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/cost_effectiveness_acceptability_frontier.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_evpi(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create separate expected value of perfect information plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Expected Value of Perfect Information: Comparison Across Interventions\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    for model_name in model_names:
        psa_results = all_results[model_name] # This is the DataFrame from psa.run_sa()
        
        # Calculate EVPI for each WTP threshold
        evpi_values = [calculate_evpi(psa_results, wtp_threshold=wtp) for wtp in wtp_thresholds]
        
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
    plt.savefig(
        f"{output_dir}/expected_value_perfect_information.png", bbox_inches="tight"
    )
    plt.savefig(
        f"{output_dir}/expected_value_perfect_information.pdf", bbox_inches="tight"
    )
    plt.close()


def plot_net_benefit_curves(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create net benefit curves with confidence intervals.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300) # Only one subplot for societal
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
            nmb_sc_iter = (psa_results['qaly_sc'] * wtp) - psa_results['cost_sc']
            nmb_nt_iter = (psa_results['qaly_nt'] * wtp) - psa_results['cost_nt']
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
    plt.savefig(f"{output_dir}/net_benefit_curves.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/net_benefit_curves.pdf", bbox_inches="tight")
    plt.close()


def plot_value_of_perspective(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create a plot showing probability of cost-effectiveness from societal perspective.
    Temporarily simplified as only societal perspective PSA is currently available.
    """
    os.makedirs(output_dir, exist_ok=True)

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
        nmb_sc_50k = (psa_results['qaly_sc'] * wtp_50k) - psa_results['cost_sc']
        nmb_nt_50k = (psa_results['qaly_nt'] * wtp_50k) - psa_results['cost_nt']
        inc_nmb_50k = nmb_nt_50k - nmb_sc_50k
        
        prob_ce = np.mean(inc_nmb_50k > 0)
        prob_cost_effective_at_50k.append(prob_ce)

    ax.bar(model_names, prob_cost_effective_at_50k, color=["blue", "green", "red"], alpha=0.7)
    ax.set_ylabel("Probability of Cost-Effectiveness", fontsize=12)
    ax.set_title(f"Probability of Cost-Effective Decision at WTP = ${wtp_50k:,.0f}/QALY")
    ax.set_ylim(0, 1) # Probability is between 0 and 1
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/value_of_perspective.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/value_of_perspective.pdf", bbox_inches="tight")
    plt.close()


def plot_pop_evpi(all_results, wtp_thresholds, output_dir="output/figures/"):
    """
    Create population EVPI plots for the societal perspective.
    Temporarily simplified as only societal perspective PSA is currently available.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Assume population sizes (these would need to be adjusted based on actual data)
    population_sizes = {
        "HPV Vaccination": 100000,  # Annual cohort for HPV vaccination
        "Smoking Cessation": 500000,  # Adult smokers in NZ
        "Hepatitis C Therapy": 50000,  # Chronic HCV cases in NZ
    }

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Population Expected Value of Perfect Information (Societal Perspective)\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    model_names = list(all_results.keys())
    for model_name in model_names:
        psa_results = all_results[model_name]
        pop_size = population_sizes.get(model_name, 1) # Default to 1 to avoid division by zero
        
        # Calculate EVPI per person for each WTP threshold
        evpi_per_person_values = [calculate_evpi(psa_results, wtp_threshold=wtp) for wtp in wtp_thresholds]
        
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
    plt.savefig(f"{output_dir}/population_evpi.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/population_evpi.pdf", bbox_inches="tight")
    plt.close()


def plot_evppi(voi_report, output_dir="output/figures/"):
    """
    Plot Expected Value of Partial Perfect Information curves.
    """
    os.makedirs(output_dir, exist_ok=True)

    evppi_data = voi_report['value_of_information']['evppi_by_parameter_group']
    wtp_thresholds = voi_report['value_of_information']['wtp_thresholds']

    if not evppi_data:
        print("No EVPPI data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.suptitle(
        "Expected Value of Partial Perfect Information (Societal Perspective)\n(2024 NZD equivalent)",
        fontsize=14,
        fontweight="bold",
    )

    for param_group, evppi_values in evppi_data.items():
        ax.plot(wtp_thresholds, evppi_values, label=param_group.replace('EVPPI_', '').replace('_', ' ').title(), linewidth=2)

    ax.set_xlabel("Willingness-To-Pay Threshold ($/QALY)", fontsize=12)
    ax.set_ylabel("EVPPI ($)", fontsize=12)
    ax.set_title("EVPPI by Parameter Group (Societal Perspective)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/expected_value_partial_perfect_information.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/expected_value_partial_perfect_information.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_comparative_two_way_dsa(comparative_results, output_dir="output/figures/"):
    """
    Create plots for comparative two-way DSA results.
    """
    os.makedirs(output_dir, exist_ok=True)

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
        plt.savefig(
            f"{output_dir}/comparative_two_way_dsa_{comparison_name.replace(' ', '_').replace('vs', 'vs')}.png",
            bbox_inches="tight",
        )
        plt.savefig(
            f"{output_dir}/comparative_two_way_dsa_{comparison_name.replace(' ', '_').replace('vs', 'vs')}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_comparative_three_way_dsa(comparative_results, output_dir="output/figures/"):
    """
    Create plots for comparative three-way DSA results.
    """
    os.makedirs(output_dir, exist_ok=True)

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
        plt.savefig(
            f"{output_dir}/comparative_three_way_dsa_{comparison_name.replace(' ', '_').replace('vs', 'vs')}.png",
            bbox_inches="tight",
        )
        plt.savefig(
            f"{output_dir}/comparative_three_way_dsa_{comparison_name.replace(' ', '_').replace('vs', 'vs')}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_cluster_analysis(cluster_results, output_dir="output/figures/"):
    """
    Create comprehensive cluster analysis visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    for intervention_name, results in cluster_results.items():
        features_pca = results["features_pca"]
        cluster_labels = results["cluster_labels"]
        n_clusters = results["n_clusters"]
        cluster_analysis = results["cluster_analysis"]

        # Create figure with subplots
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

        ax.set_xlabel(f'PC1 ({results["pca_explained_variance"][0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({results["pca_explained_variance"][1]*100:.1f}% variance)')
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
            means = cluster_analysis[f"cluster_{cluster_id}"]["means"][:6]  # First 6 features
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
        plt.savefig(
            f"{output_dir}/cluster_analysis_{intervention_name.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )
        plt.savefig(
            f"{output_dir}/cluster_analysis_{intervention_name.lower().replace(' ', '_')}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_comparative_clusters(cluster_results, output_dir="output/figures/"):
    """
    Create comparative cluster analysis across all interventions.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not cluster_results:
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
            size = results["cluster_analysis"][f"cluster_{cluster_id}"]["size"]
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
    for i, data in enumerate(archetype_data):
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
    plt.savefig(f"{output_dir}/comparative_cluster_analysis.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/comparative_cluster_analysis.pdf", bbox_inches="tight")
    plt.close()