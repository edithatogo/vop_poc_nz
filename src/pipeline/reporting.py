"""
Reporting Pipeline Module.

This module handles the generation of all figures, dashboards, and reports
based on the results from the analysis pipeline.
"""

import logging
import os

import numpy as np
import pandas as pd

from ..dcea_equity_analysis import (
    plot_combined_lorenz_curves,
    plot_comparative_equity_impact_plane,
    plot_probabilistic_equity_impact_plane,
    plot_probabilistic_equity_impact_plane_with_delta,
)
from ..dsa_analysis import (
    compose_dsa_dashboard,
    plot_one_way_dsa_tornado,
    plot_three_way_dsa_3d,
    plot_two_way_dsa_heatmaps,
)
from ..policy_brief_generator import generate_policy_brief
from ..table_generation import generate_all_tables
from ..visualizations import (
    compose_dashboard,
    plot_annual_cash_flow,
    plot_ceac,
    plot_ceaf,
    plot_comparative_bia_line,
    plot_comparative_ce_plane,
    plot_comparative_ce_plane_with_delta,
    plot_comparative_ceac,
    plot_comparative_ceac_with_delta,
    plot_comparative_evpi,
    plot_comparative_evpi_with_delta,
    plot_comparative_evppi,
    plot_comparative_evppi_with_delta,
    plot_comparative_pop_evpi_with_delta,
    plot_cost_effectiveness_plane,
    plot_decision_tree,
    plot_discordance_loss,
    plot_inequality_aversion_sensitivity,
    plot_markov_trace,
    plot_net_benefit_curves,
    plot_pop_evpi,
    plot_value_of_perspective,
)

logger = logging.getLogger(__name__)


def run_reporting_pipeline(results: dict, output_dir: str = "output"):
    """
    Run the complete reporting pipeline.
    Generates all figures, dashboards, and the policy brief.
    """
    logger.info("=" * 70)
    print("DEBUG: run_reporting_pipeline EXECUTED")
    logger.info("RUNNING REPORTING PIPELINE")
    logger.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 0. Save Full Results to JSON (for debugging and reproducibility)
    print("\nSaving full analysis results to JSON...")
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="list")
            if isinstance(obj, pd.Series):
                return obj.to_list()
            return super().default(obj)

    try:
        with open(os.path.join(output_dir, "complete_analysis_results.json"), "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        print(f"Results saved to {os.path.join(output_dir, 'complete_analysis_results.json')}")
    except Exception as e:
        print(f"Warning: Could not save results to JSON: {e}")

    # 1. Write Text Reports
    logger.info("Writing results to output files...")
    for name, report in results["reports"].items():
        with open(os.path.join(output_dir, f"{name}_report.md"), "w") as f:
            f.write(report)

    # Combine reports
    with open(os.path.join(output_dir, "combined_report.md"), "w") as outfile:
        # Start with Policy Brief (will be generated later, but we can structure it here or append)
        # Actually, let's just combine the intervention reports for now
        outfile.write("# Combined Analysis Report\n\n")
        for name, report in results["reports"].items():
            outfile.write(f"\n\n---\n\n# {name}\n\n")
            outfile.write(report)

    # 1.5 Generate Tables
    logger.info("Generating tables...")
    generate_all_tables(results, results.get("selected_interventions", {}), os.path.join(output_dir, "tables"))

    # 2. Generate Plots
    logger.info("Generating all plots...")
    wtp_thresholds = np.linspace(0, 100000, 21)

    # CE Plane & Curves
    plot_cost_effectiveness_plane(
        results["probabilistic_results"], perspective="societal", output_dir=figures_dir
    )
    plot_comparative_ce_plane(
        results["intervention_results"],
        output_dir=figures_dir,
        psa_results=results.get("probabilistic_results"),
    )
    plot_ceac(
        results["probabilistic_results"],
        wtp_thresholds,
        perspective="societal",
        output_dir=figures_dir,
    )
    plot_comparative_ceac(
        results["probabilistic_results"],
        output_dir=figures_dir,
    )
    plot_ceaf(
        results["probabilistic_results"],
        wtp_thresholds,
        perspective="societal",
        output_dir=figures_dir,
    )
    plot_comparative_evpi(
        results["probabilistic_results"],
        wtp_thresholds,
        output_dir=figures_dir,
    )
    plot_net_benefit_curves(
        results["probabilistic_results"],
        wtp_thresholds,
        perspective="societal",
        output_dir=figures_dir,
    )
    plot_value_of_perspective(
        results["probabilistic_results"],
        wtp_thresholds,
        perspective="societal",
        output_dir=figures_dir,
    )
    # Extract population sizes for EVPI plot
    population_sizes = {}
    for name, params in results.get("selected_interventions", {}).items():
        bia_pop = params.get("bia_population", {})
        population_sizes[name] = bia_pop.get("total_population", 100000)

    plot_pop_evpi(
        results["probabilistic_results"],
        wtp_thresholds,
        population_sizes,
        perspective="societal",
        output_dir=figures_dir,
    )
    plot_comparative_evppi(
        results["voi_analysis"],
        output_dir=figures_dir,
    )
    # Comparative Population EVPI
    logger.info("Generating Comparative Population EVPI...")
    plot_comparative_pop_evpi_with_delta(
        results["probabilistic_results"],
        wtp_thresholds,
        population_sizes,
        output_dir=figures_dir,
    )

    # Delta Plots (New Request)
    plot_comparative_ce_plane_with_delta(
        results["probabilistic_results"],
        output_dir=figures_dir,
    )
    plot_comparative_ceac_with_delta(
        results["probabilistic_results"],
        wtp_thresholds,
        output_dir=figures_dir,
    )
    plot_comparative_evpi_with_delta(
        results["probabilistic_results"],
        wtp_thresholds,
        output_dir=figures_dir,
    )
    plot_comparative_evppi_with_delta(
        results["voi_analysis"],
        output_dir=figures_dir,
    )
    # Decision Trees
    for name, params in results["selected_interventions"].items():
        plot_decision_tree(name, params, output_dir=figures_dir)

    # DSA Plots
    plot_one_way_dsa_tornado(results["dsa_analysis"]["1_way"], output_dir=figures_dir)
    plot_two_way_dsa_heatmaps(results["dsa_analysis"]["2_way"], output_dir=figures_dir)
    plot_three_way_dsa_3d(results["dsa_analysis"]["3_way"], output_dir=figures_dir)
    compose_dsa_dashboard(output_dir=figures_dir)

    # BIA Plots
    for name, bia_df in results["bia_results"].items():
        plot_annual_cash_flow(
            years=bia_df["year"].tolist(),
            gross_costs=bia_df["gross_cost"].tolist(),
            net_costs=bia_df["net_cost"].tolist(),
            output_dir=figures_dir,
            intervention=name,
        )
    # compose_bia_dashboard(output_dir=figures_dir)
    # compose_bia_dashboard(output_dir=figures_dir)
    plot_comparative_bia_line(results["bia_results"], output_dir=figures_dir)

    # Markov Traces
    logger.info("Generating Markov Traces...")
    for name, res in results["intervention_results"].items():
        logger.info(f"Checking traces for {name}...")
        if "health_system" in res:
            hs_res = res["health_system"]
            trace_data = {}
            if "trace_standard_care" in hs_res and hs_res["trace_standard_care"] is not None:
                logger.info(f"  Found trace_standard_care for {name}")
                # Convert to DataFrame for plotting
                # Need state names from params
                params = results["selected_interventions"].get(name, {})
                states = params.get("states", [])
                if states:
                    trace_data["standard_care"] = pd.DataFrame(hs_res["trace_standard_care"], columns=states)
                else:
                    logger.warning(f"  States missing for {name}")
            else:
                logger.warning(f"  trace_standard_care missing or None for {name}")

            if "trace_new_treatment" in hs_res and hs_res["trace_new_treatment"] is not None:
                logger.info(f"  Found trace_new_treatment for {name}")
                params = results["selected_interventions"].get(name, {})
                states = params.get("states", [])
                if states:
                    trace_data["new_treatment"] = pd.DataFrame(hs_res["trace_new_treatment"], columns=states)
            else:
                logger.warning(f"  trace_new_treatment missing or None for {name}")

            if trace_data:
                logger.info(f"  Plotting trace for {name}")
                plot_markov_trace(trace_data, figures_dir, name)
            else:
                logger.warning(f"  No trace data collected for {name}")

    # Equity Plots
    equity_interventions = []
    for name, dcea_res in results.get("dcea_equity_analysis", {}).items():
        if dcea_res:
            pass # Added to fix IndentationError
    if equity_interventions:
        # compose_equity_dashboard(equity_interventions, output_dir=figures_dir)
        pass # Added to fix IndentationError

    # Comparative Equity Plot
    if results.get("dcea_equity_analysis"):
        plot_comparative_equity_impact_plane(
            results["dcea_equity_analysis"],
            output_dir=figures_dir
        )

    # Probabilistic Equity Plot (Scatter)
    if results.get("probabilistic_results"):
        plot_probabilistic_equity_impact_plane(
            results["probabilistic_results"],
            output_dir=figures_dir
        )
        plot_probabilistic_equity_impact_plane_with_delta(
            results["probabilistic_results"],
            output_dir=figures_dir
        )

    # Combined Lorenz Curves
    if results.get("dcea_equity_analysis"):
        plot_combined_lorenz_curves(
            results["dcea_equity_analysis"],
            output_dir=figures_dir
        )

    # Inequality Aversion Sensitivity Plots
    for name, res in results["intervention_results"].items():
        if "societal" in res and "human_capital" in res["societal"]:
            sensitivity_data = res["societal"]["human_capital"].get(
                "inequality_sensitivity"
            )
            if sensitivity_data is not None and not sensitivity_data.empty:
                plot_inequality_aversion_sensitivity(
                    sensitivity_data, name, output_dir=figures_dir
                )

    # Discordance Loss Plot
    discordance_data = []
    for _name, res in results["intervention_results"].items():
        if "discordance" in res:
            discordance_data.append(res["discordance"])
    if discordance_data:
        plot_discordance_loss(discordance_data, output_dir=figures_dir)

    # 3. Dashboards
    dashboard_images = [
        os.path.join(figures_dir, "cost_effectiveness_plane_societal.png"),
        os.path.join(
            figures_dir, "cost_effectiveness_acceptability_curve_societal.png"
        ),
        os.path.join(
            figures_dir, "cost_effectiveness_acceptability_frontier_societal.png"
        ),
        os.path.join(figures_dir, "expected_value_perfect_information_societal.png"),
        os.path.join(figures_dir, "net_benefit_curves_societal.png"),
        os.path.join(figures_dir, "value_of_perspective_societal.png"),
        os.path.join(figures_dir, "population_evpi_societal.png"),
        os.path.join(
            figures_dir, "expected_value_partial_perfect_information_societal.png"
        ),
    ]
    compose_dashboard(
        dashboard_images,
        output_dir=figures_dir,
        filename_base="dashboard_ce_voi_societal",
    )

    # 4. Policy Brief
    logger.info("Generating Policy Brief...")
    generate_policy_brief(results["intervention_results"], output_dir=reports_dir)

    logger.info(f"Reporting complete. Outputs saved to {output_dir}/")
