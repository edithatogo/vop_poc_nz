import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from .cea_model_core import run_cea
from .visualizations import apply_default_style, build_filename_base, save_figure

logger = logging.getLogger(__name__)


def perform_one_way_dsa(models, wtp_threshold=50000, n_points=20):
    """
    Perform one-way DSA for all interventions.
    """
    results = {}

    for model_name, model_params in models.items():
        logger.info(f"Performing one-way DSA for {model_name}...")

        # Get parameter ranges from YAML or use defaults
        yaml_ranges = model_params.get("dsa_parameter_ranges", {})
        param_ranges = {}

        if yaml_ranges:
            # Use ranges from YAML
            for param, data in yaml_ranges.items():
                # Create range from min/max with n_points
                min_val, max_val = data["range"]
                param_ranges[param] = np.linspace(min_val, max_val, n_points)
        else:
            # Fallback defaults
            param_ranges = {
                "cost_multiplier": np.linspace(0.5, 2.0, n_points),
                "qaly_multiplier": np.linspace(0.5, 1.5, n_points),
                "discount_rate": np.linspace(0.0, 0.10, n_points),
                "wtp_threshold": np.linspace(25000, 75000, n_points),
            }

        dsa_results = {}

        # Calculate base NMB
        base_hs_results = run_cea(
            model_params, perspective="health_system", wtp_threshold=wtp_threshold
        )
        base_nmb_hs = (
            base_hs_results["incremental_qalys"] * wtp_threshold
        ) - base_hs_results["incremental_cost"]
        base_soc_results = run_cea(
            model_params, perspective="societal", wtp_threshold=wtp_threshold
        )
        base_nmb_soc = (
            base_soc_results["incremental_qalys"] * wtp_threshold
        ) - base_soc_results["incremental_cost"]

        for param_name, param_range in param_ranges.items():
            param_results_hs = []
            param_results_soc = []

            for param_value in param_range:
                temp_params = copy.deepcopy(model_params)
                current_wtp = wtp_threshold  # Default WTP

                # Apply parameter change
                if param_name == "wtp_threshold":
                    current_wtp = param_value
                elif param_name == "cost_multiplier":
                    temp_params["costs"]["health_system"]["new_treatment"][0] *= (
                        param_value
                    )
                    temp_params["costs"]["societal"]["new_treatment"][0] *= param_value
                elif param_name == "qaly_multiplier":
                    if (
                        "qalys" in temp_params
                        and len(temp_params["qalys"]["new_treatment"]) > 1
                    ):
                        temp_params["qalys"]["new_treatment"][1] *= param_value
                elif param_name == "discount_rate":
                    temp_params["discount_rate"] = param_value

                # Extended parameters
                elif param_name == "transition_healthy_to_sick":
                    # Update [0][1] and adjust [0][0] to maintain sum=1
                    for arm in ["standard_care", "new_treatment"]:
                        if arm in temp_params["transition_matrices"]:
                            matrix = temp_params["transition_matrices"][arm]
                            diff = param_value - matrix[0][1]
                            matrix[0][1] = param_value
                            matrix[0][0] -= diff  # Adjust healthy->healthy

                elif param_name == "transition_sick_to_dead":
                    # Update [1][2] and adjust [1][1]
                    for arm in ["standard_care", "new_treatment"]:
                        if arm in temp_params["transition_matrices"]:
                            matrix = temp_params["transition_matrices"][arm]
                            diff = param_value - matrix[1][2]
                            matrix[1][2] = param_value
                            matrix[1][1] -= diff  # Adjust sick->sick

                elif param_name == "cost_sick_hs":
                    # Update cost index 1 (Sick)
                    for arm in ["standard_care", "new_treatment"]:
                        if arm in temp_params["costs"]["health_system"]:
                            temp_params["costs"]["health_system"][arm][1] = param_value

                elif param_name == "cost_sick_societal":
                    for arm in ["standard_care", "new_treatment"]:
                        if arm in temp_params["costs"]["societal"]:
                            temp_params["costs"]["societal"][arm][1] = param_value

                elif param_name == "qaly_sick":
                    for arm in ["standard_care", "new_treatment"]:
                        if arm in temp_params["qalys"]:
                            temp_params["qalys"][arm][1] = param_value

                elif param_name == "productivity_cost_sick":
                    if (
                        "productivity_costs" in temp_params
                        and "human_capital" in temp_params["productivity_costs"]
                    ):
                        for arm in ["standard_care", "new_treatment"]:
                            if (
                                arm
                                in temp_params["productivity_costs"]["human_capital"]
                            ):
                                temp_params["productivity_costs"]["human_capital"][arm][
                                    1
                                ] = param_value

                elif param_name == "friction_period_days":
                    if "friction_cost_params" in temp_params:
                        temp_params["friction_cost_params"]["friction_period_days"] = (
                            param_value
                        )

                # Health system perspective
                hs_results = run_cea(
                    temp_params,
                    perspective="health_system",
                    wtp_threshold=current_wtp,
                )
                nmb_hs = (hs_results["incremental_qalys"] * current_wtp) - hs_results[
                    "incremental_cost"
                ]
                param_results_hs.append(nmb_hs)

                # Societal perspective
                soc_results = run_cea(
                    temp_params, perspective="societal", wtp_threshold=current_wtp
                )
                nmb_soc = (
                    soc_results["incremental_qalys"] * current_wtp
                ) - soc_results["incremental_cost"]
                param_results_soc.append(nmb_soc)

            dsa_results[param_name] = {
                "param_range": param_range,
                "nmb_hs": param_results_hs,
                "nmb_soc": param_results_soc,
            }

        results[model_name] = {
            "dsa_results": dsa_results,
            "base_nmb_hs": base_nmb_hs,
            "base_nmb_soc": base_nmb_soc,
        }

    return results


def plot_one_way_dsa_tornado(dsa_results, output_dir="data/data_outputs/figures/"):
    """
    Create a combined one-way DSA tornado plot (2 rows x N columns).
    Row 1: Health System
    Row 2: Societal
    """
    apply_default_style()

    if not dsa_results:
        logger.warning("No one-way DSA results to plot.")
        return

    model_names = list(dsa_results.keys())
    n_models = len(model_names)
    
    # Create figure: 2 rows, n_models columns
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 12), dpi=300, sharey=False)
    
    # Ensure axes is always 2D array (2, n_models)
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_models > 1:
        # axes is already (2, n_models)
        pass
    
    perspectives = ["hs", "soc"]
    perspective_titles = ["Health System", "Societal"]
    
    import matplotlib.ticker as mtick

    for col, model_name in enumerate(model_names):
        results = dsa_results[model_name]
        
        for row, perspective in enumerate(perspectives):
            ax = axes[row, col]
            
            base_nmb = results[f"base_nmb_{perspective}"]
            
            param_names = []
            nmb_ranges = []
            
            for param_name, param_results in results["dsa_results"].items():
                param_names.append(param_name)
                nmb_range = np.array(param_results[f"nmb_{perspective}"])
                nmb_ranges.append(nmb_range)
            
            # Sort by range width
            range_widths = [np.max(r) - np.min(r) for r in nmb_ranges]
            sorted_indices = np.argsort(range_widths)[::-1]
            
            sorted_param_names = [param_names[i] for i in sorted_indices]
            sorted_nmb_ranges = [nmb_ranges[i] for i in sorted_indices]
            
            y_pos = np.arange(len(sorted_param_names))
            
            for i, nmb_range in enumerate(sorted_nmb_ranges):
                min_nmb = np.min(nmb_range)
                max_nmb = np.max(nmb_range)
                ax.barh(
                    y_pos[i],
                    max_nmb - min_nmb,
                    left=min_nmb,
                    color="skyblue",
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5
                )
            
            ax.axvline(base_nmb, color="black", linestyle="--", linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_param_names, fontsize=8)
            
            # Only set ylabel for the first column
            # if col == 0:
            #     ax.set_ylabel("Parameter", fontsize=10)
            
            ax.set_xlabel("Net Monetary Benefit ($)", fontsize=10)
            
            # Title for each subplot
            if row == 0:
                ax.set_title(f"{model_name}\n({perspective_titles[row]})", fontsize=12, fontweight="bold")
            else:
                ax.set_title(f"({perspective_titles[row]})", fontsize=12)
                
            ax.grid(True, alpha=0.3)
            
            # Format x-axis as currency
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        "comparative_one_way_dsa_tornado",
    )


def perform_comprehensive_two_way_dsa(  # noqa: C901
    models, wtp_threshold=50000, n_points=20
):
    """
    Perform comprehensive two-way DSA for all interventions.
    """
    results = {}

    for model_name, model_params in models.items():
        logger.info(f"Performing two-way DSA for {model_name}...")

        # Define parameter ranges based on intervention type
        if "HPV" in model_name:
            param1_name = "Vaccine Cost Multiplier"
            param2_name = "Cancer Treatment Cost Reduction"
            param1_range = np.linspace(0.5, 2.0, n_points)
            param2_range = np.linspace(0.5, 1.5, n_points)
        elif "Smoking" in model_name:
            param1_name = "Intervention Cost Multiplier"
            param2_name = "Productivity Cost Multiplier"
            param1_range = np.linspace(0.5, 2.0, n_points)
            param2_range = np.linspace(0.5, 1.5, n_points)
        elif "Hepatitis C" in model_name:
            param1_name = "Treatment Cost Multiplier"
            param2_name = "QALY Improvement Multiplier"
            param1_range = np.linspace(0.5, 2.5, n_points)
            param2_range = np.linspace(0.5, 1.5, n_points)
        elif "Obesity" in model_name:
            param1_name = "Intervention Cost Multiplier"
            param2_name = "Societal Benefit Multiplier"
            param1_range = np.linspace(0.5, 2.0, n_points)
            param2_range = np.linspace(0.5, 1.5, n_points)
        elif "Housing" in model_name:
            param1_name = "Insulation Cost Multiplier"
            param2_name = "Energy Savings Multiplier"
            param1_range = np.linspace(0.5, 2.0, n_points)
            param2_range = np.linspace(0.5, 1.5, n_points)
        else:  # Cancer drug
            param1_name = "Drug Cost Multiplier"
            param2_name = "QALY Gain Multiplier"
            param1_range = np.linspace(0.5, 2.5, n_points)
            param2_range = np.linspace(0.5, 1.5, n_points)

        dsa_grid_hs = []
        dsa_grid_soc = []

        for p1 in param1_range:
            for p2 in param2_range:
                temp_params = copy.deepcopy(model_params)

                if "HPV" in model_name:
                    temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                    temp_params["costs"]["health_system"]["new_treatment"][1] *= p2
                elif "Smoking" in model_name:
                    temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                    temp_params["costs"]["societal"]["new_treatment"][1] *= p2
                elif "Hepatitis C" in model_name:
                    temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                    temp_params["qalys"]["new_treatment"][1] *= p2
                elif "Obesity" in model_name or "Housing" in model_name:
                    temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                    temp_params["costs"]["societal"]["new_treatment"][1] *= p2
                else:  # Cancer drug
                    temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                    temp_params["qalys"]["new_treatment"][1] *= p2

                # Health system perspective
                hs_results = run_cea(
                    temp_params,
                    perspective="health_system",
                    wtp_threshold=wtp_threshold,
                )
                nmb_hs = (hs_results["incremental_qalys"] * wtp_threshold) - hs_results[
                    "incremental_cost"
                ]
                dsa_grid_hs.append(
                    {
                        "param1": p1,
                        "param2": p2,
                        "nmb": nmb_hs,
                        "icer": hs_results["icer"],
                    }
                )

                # Societal perspective
                soc_results = run_cea(
                    temp_params, perspective="societal", wtp_threshold=wtp_threshold
                )
                nmb_soc = (
                    soc_results["incremental_qalys"] * wtp_threshold
                ) - soc_results["incremental_cost"]
                dsa_grid_soc.append(
                    {
                        "param1": p1,
                        "param2": p2,
                        "nmb": nmb_soc,
                        "icer": soc_results["icer"],
                    }
                )

        results[model_name] = {
            "param1_name": param1_name,
            "param2_name": param2_name,
            "param1_range": param1_range,
            "param2_range": param2_range,
            "dsa_grid_hs": dsa_grid_hs,
            "dsa_grid_soc": dsa_grid_soc,
        }

    return results


def plot_two_way_dsa_heatmaps(dsa_results, output_dir="data/data_outputs/figures/"):
    """
    Create a combined two-way DSA heatmap plot (2 rows x N columns).
    Row 1: Health System
    Row 2: Societal
    """
    apply_default_style()

    if not dsa_results:
        logger.warning("No two-way DSA results to plot.")
        return

    model_names = list(dsa_results.keys())
    n_models = len(model_names)
    
    # Create figure: 2 rows, n_models columns
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10), dpi=300)
    
    # Ensure axes is always 2D array (2, n_models)
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_models > 1:
        # axes is already (2, n_models)
        pass
        
    perspectives = ["hs", "soc"]
    perspective_titles = ["Health System", "Societal"]

    for col, model_name in enumerate(model_names):
        data = dsa_results[model_name]
        param1_range = np.array(data["param1_range"])
        param2_range = np.array(data["param2_range"])
        
        for row, perspective in enumerate(perspectives):
            ax = axes[row, col]
            grid_key = f"dsa_grid_{perspective}"

            # Create grid matrices
            nmb_grid = np.zeros((len(param1_range), len(param2_range)))
            for point in data[grid_key]:
                p1_idx = np.argmin(np.abs(param1_range - point["param1"]))
                p2_idx = np.argmin(np.abs(param2_range - point["param2"]))
                nmb_grid[p1_idx, p2_idx] = point["nmb"]

            # Plot heatmap
            im = ax.imshow(nmb_grid, cmap="RdYlGn", origin="lower", aspect="auto")
            
            # Axis labels
            ax.set_xlabel(data["param2_name"], fontsize=9)
            # Only set ylabel for the first column
            if col == 0:
                ax.set_ylabel(data["param1_name"], fontsize=9)
            
            # Title
            if row == 0:
                ax.set_title(f"{model_name}\n({perspective_titles[row]})", fontsize=11, fontweight="bold")
            else:
                ax.set_title(f"({perspective_titles[row]})", fontsize=11)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("NMB ($)", fontsize=8)

            # Set tick labels (simplified for readability)
            # X-axis
            x_ticks = np.linspace(0, len(param2_range)-1, 5, dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{param2_range[i]:.1f}" for i in x_ticks], fontsize=8, rotation=45)
            
            # Y-axis
            y_ticks = np.linspace(0, len(param1_range)-1, 5, dtype=int)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{param1_range[i]:.1f}" for i in y_ticks], fontsize=8)

    plt.tight_layout()
    save_figure(
        fig,
        output_dir,
        "comparative_two_way_dsa_heatmaps",
    )


def compose_dsa_dashboard(output_dir="data/data_outputs/figures/"):
    """Compose a basic DSA dashboard from generated PNGs."""
    from glob import glob

    images = []
    # collect one-way plots (may include model/perspective tokens)
    images.extend(sorted(glob(os.path.join(output_dir, "one_way_dsa_tornado*.png"))))
    # two-way (perspective-level)
    images.extend(sorted(glob(os.path.join(output_dir, "two_way_dsa_heatmaps_*.png"))))
    # three-way (may include model token)
    images.extend(sorted(glob(os.path.join(output_dir, "three_way_dsa_3d*.png"))))
    from .visualizations import compose_dashboard

    compose_dashboard(images, output_dir=output_dir, filename_base="dashboard_dsa")


def perform_three_way_dsa(  # noqa: C901
    models, wtp_threshold=50000, n_points=10
):
    """
    Perform three-way deterministic sensitivity analysis.
    """
    results = {}

    for model_name, model_params in models.items():
        logger.info(f"Performing three-way DSA for {model_name}...")

        if "HPV" in model_name:
            param1_name = "Vaccine Cost Multiplier"
            param2_name = "Cancer Treatment Cost Multiplier"
            param3_name = "QALY Improvement Multiplier"
            p1_range = np.linspace(0.5, 2.0, n_points)
            p2_range = np.linspace(0.5, 2.0, n_points)
            p3_range = np.linspace(0.5, 1.5, n_points)
        elif "Smoking" in model_name:
            param1_name = "Intervention Cost Multiplier"
            param2_name = "Productivity Cost Multiplier"
            param3_name = "Success Rate Multiplier"
            p1_range = np.linspace(0.5, 2.0, n_points)
            p2_range = np.linspace(0.5, 1.5, n_points)
            p3_range = np.linspace(0.8, 1.2, n_points)
        elif "Hepatitis C" in model_name:
            param1_name = "Treatment Cost Multiplier"
            param2_name = "QALY Improvement Multiplier"
            param3_name = "Duration Multiplier"
            p1_range = np.linspace(0.5, 2.0, n_points)
            p2_range = np.linspace(0.8, 1.2, n_points)
            p3_range = np.linspace(0.8, 1.2, n_points)
        elif "Obesity" in model_name:
            param1_name = "Intervention Cost Multiplier"
            param2_name = "Societal Benefit Multiplier"
            param3_name = "QALY Improvement Multiplier"
            p1_range = np.linspace(0.5, 2.0, n_points)
            p2_range = np.linspace(0.5, 1.5, n_points)
            p3_range = np.linspace(0.8, 1.2, n_points)
        elif "Housing" in model_name:
            param1_name = "Insulation Cost Multiplier"
            param2_name = "Energy Savings Multiplier"
            param3_name = "Health Benefit Multiplier"
            p1_range = np.linspace(0.5, 2.0, n_points)
            p2_range = np.linspace(0.5, 1.5, n_points)
            p3_range = np.linspace(0.8, 1.2, n_points)
        else:  # Cancer drug
            param1_name = "Drug Cost Multiplier"
            param2_name = "QALY Gain Multiplier"
            param3_name = "Transition Prob Multiplier"
            p1_range = np.linspace(0.5, 2.0, n_points)
            p2_range = np.linspace(0.8, 1.2, n_points)
            p3_range = np.linspace(0.8, 1.2, n_points)

        dsa_grid = []

        for p1 in p1_range:
            for p2 in p2_range:
                for p3 in p3_range:
                    temp_params = copy.deepcopy(model_params)

                    if "HPV" in model_name:
                        temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                        temp_params["costs"]["health_system"]["new_treatment"][1] *= p2
                        temp_params["qalys"]["new_treatment"][1] *= p3
                    elif "Smoking" in model_name:
                        temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                        temp_params["costs"]["societal"]["new_treatment"][1] *= p2
                        temp_params["transition_matrices"]["new_treatment"][0][1] *= p3
                        temp_params["transition_matrices"]["new_treatment"][0][0] = (
                            1
                            - temp_params["transition_matrices"]["new_treatment"][0][1]
                            - temp_params["transition_matrices"]["new_treatment"][0][2]
                        )
                    elif "Hepatitis C" in model_name:
                        temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                        temp_params["qalys"]["new_treatment"][1] *= p2
                        temp_params["cycles"] = int(model_params["cycles"] * p3)
                    elif "Obesity" in model_name or "Housing" in model_name:
                        temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                        temp_params["costs"]["societal"]["new_treatment"][1] *= p2
                        temp_params["qalys"]["new_treatment"][1] *= p3
                    else:  # Cancer drug
                        temp_params["costs"]["health_system"]["new_treatment"][0] *= p1
                        temp_params["qalys"]["new_treatment"][1] *= p2
                        temp_params["transition_matrices"]["new_treatment"][0][1] *= p3
                        temp_params["transition_matrices"]["new_treatment"][0][0] = (
                            1
                            - temp_params["transition_matrices"]["new_treatment"][0][1]
                            - temp_params["transition_matrices"]["new_treatment"][0][2]
                        )

                    soc_results = run_cea(
                        temp_params, perspective="societal", wtp_threshold=wtp_threshold
                    )
                    nmb = (
                        soc_results["incremental_qalys"] * wtp_threshold
                    ) - soc_results["incremental_cost"]
                    dsa_grid.append(
                        {
                            "p1": p1,
                            "p2": p2,
                            "p3": p3,
                            "nmb": nmb,
                            "icer": soc_results["icer"],
                        }
                    )

        results[model_name] = {
            "param1_name": param1_name,
            "param2_name": param2_name,
            "param3_name": param3_name,
            "p1_range": p1_range,
            "p2_range": p2_range,
            "p3_range": p3_range,
            "dsa_grid": dsa_grid,
        }

    return results


def plot_three_way_dsa_3d(dsa_results, output_dir="data/data_outputs/figures/"):
    """
    Create 3D surface plots for three-way DSA.
    """
    apply_default_style()

    model_names = list(dsa_results.keys())
    if not model_names:
        logger.warning("No three-way DSA results to plot.")
        return

    for model_name in model_names:
        data = dsa_results[model_name]

        # Create 3D grid
        p1_grid, p2_grid, p3_grid = np.meshgrid(
            data["p1_range"], data["p2_range"], data["p3_range"], indexing="ij"
        )

        # Interpolate NMB values onto the grid
        points = np.array(
            [[point["p1"], point["p2"], point["p3"]] for point in data["dsa_grid"]]
        )
        values = np.array([point["nmb"] for point in data["dsa_grid"]])

        # Create slices at different levels of p3 (min, median, max)
        p3_min, p3_max = np.min(data["p3_range"]), np.max(data["p3_range"])
        p3_slices = [p3_min, (p3_min + p3_max) / 2, p3_max]

        fig = plt.figure(figsize=(5 * len(p3_slices), 5), dpi=300)
        fig.suptitle(
            f"Three-Way Deterministic Sensitivity Analysis: {model_name}\nNMB at $50,000/QALY (Societal Perspective)",
            fontsize=14,
            fontweight="bold",
        )

        for i, p3_slice in enumerate(p3_slices):
            ax = fig.add_subplot(1, len(p3_slices), i + 1, projection="3d")

            # Find points near this p3 slice
            mask = (points[:, 2] >= p3_slice - 0.05) & (points[:, 2] <= p3_slice + 0.05)
            if np.sum(mask) > 10:  # pragma: no cover - requires dense grids
                slice_points = points[mask]
                slice_values = values[mask]

                # Create 2D grid for this slice
                p1_unique = np.unique(slice_points[:, 0])
                p2_unique = np.unique(slice_points[:, 1])

                if len(p1_unique) > 1 and len(p2_unique) > 1:
                    grid_p1, grid_p2 = np.meshgrid(p1_unique, p2_unique, indexing="ij")
                    grid_nmb = griddata(
                        (slice_points[:, 0], slice_points[:, 1]),
                        slice_values,
                        (grid_p1, grid_p2),
                        method="linear",
                    )

                    surf = ax.plot_surface(
                        grid_p1, grid_p2, grid_nmb, cmap="RdYlGn", alpha=0.8
                    )
                    ax.set_xlabel(data["param1_name"][:15] + "...", fontsize=8)
                    ax.set_ylabel(data["param2_name"][:15] + "...", fontsize=8)
                    ax.set_zlabel("NMB ($)", fontsize=8)
                    ax.set_title(f"{data['param3_name']} = {p3_slice:.2f}", fontsize=10)

                    # Add colorbar
                    cbar = plt.colorbar(surf, ax=ax, shrink=0.5)
                    cbar.set_label("NMB ($)", fontsize=6)

        plt.tight_layout()
        save_figure(
            fig, output_dir, build_filename_base("three_way_dsa_3d", model_name)
        )
