import numpy as np
import pandas as pd
import pytest

from src.cluster_analysis import ClusterAnalysis
from src.dsa_analysis import (
    compose_dsa_dashboard,
    perform_comprehensive_two_way_dsa,
    perform_one_way_dsa,
    perform_three_way_dsa,
    plot_one_way_dsa_tornado,
    plot_three_way_dsa_3d,
    plot_two_way_dsa_heatmaps,
)
from src.visualizations import (
    compose_bia_dashboard,
    compose_equity_dashboard,
    plot_cluster_analysis,
    plot_comparative_clusters,
    plot_comparative_three_way_dsa,
    plot_comparative_two_way_dsa,
    plot_evp_curve,
    plot_financial_risk_protection,
    plot_inequality_aversion_sensitivity,
    plot_inequality_staircase,
    plot_societal_drivers,
    plot_structural_tornado,
    plot_survival_gof,
)


@pytest.fixture()
def base_params():
    return {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.05, 0.05], [0.0, 0.85, 0.15], [0.0, 0.0, 1.0]],
            "new_treatment": [[0.92, 0.05, 0.03], [0.0, 0.88, 0.12], [0.0, 0.0, 1.0]],
        },
        "cycles": 3,
        "initial_population": [1000, 0, 0],
        "costs": {
            "health_system": {
                "standard_care": [0, 500, 0],
                "new_treatment": [100, 450, 0],
            },
            "societal": {
                "standard_care": [0, 800, 0],
                "new_treatment": [50, 500, 0],
            },
        },
        "qalys": {"standard_care": [1.0, 0.7, 0.0], "new_treatment": [1.0, 0.8, 0.0]},
        "productivity_costs": {
            "human_capital": {
                "standard_care": [0, 200, 0],
                "new_treatment": [0, 150, 0],
            }
        },
    }


def test_cluster_analysis_and_plots(tmp_path):
    rng = np.random.default_rng(0)
    all_results = {
        "Demo": {
            "inc_cost": rng.normal(0, 1000, size=40),
            "inc_qaly": rng.normal(0.1, 0.05, size=40),
        }
    }
    analyzer = ClusterAnalysis(all_results, ["Demo"])
    cluster_res = analyzer.perform_clustering("Demo", n_clusters_range=[2, 3])

    fig_dir = tmp_path / "cluster_figs"
    fig_dir.mkdir()
    plot_cluster_analysis({"Demo": cluster_res}, output_dir=str(fig_dir))
    plot_comparative_clusters(
        {"A": cluster_res, "B": cluster_res}, output_dir=str(fig_dir)
    )
    assert any(fig_dir.iterdir())


def test_comparative_dsa_plots(tmp_path):
    fig_dir = tmp_path / "dsa_figs"
    fig_dir.mkdir()

    two_way = {
        "HPV vs Smoking": {
            "comparative_grid": [
                {"inc_nmb_hs": 10, "inc_nmb_soc": 5},
                {"inc_nmb_hs": -3, "inc_nmb_soc": -1},
            ]
        }
    }
    plot_comparative_two_way_dsa(two_way, output_dir=str(fig_dir))

    three_way = {
        "Three Way": {
            "comparative_grid": [
                {
                    "hpv_vs_smoking_nmb_hs": 10,
                    "hpv_vs_hep_c_nmb_hs": 5,
                    "smoking_vs_hep_c_nmb_hs": -2,
                    "hpv_vs_smoking_nmb_soc": 8,
                    "hpv_vs_hep_c_nmb_soc": 4,
                    "smoking_vs_hep_c_nmb_soc": -1,
                },
                {
                    "hpv_vs_smoking_nmb_hs": -1,
                    "hpv_vs_hep_c_nmb_hs": 3,
                    "smoking_vs_hep_c_nmb_hs": 1,
                    "hpv_vs_smoking_nmb_soc": 2,
                    "hpv_vs_hep_c_nmb_soc": 1,
                    "smoking_vs_hep_c_nmb_soc": -2,
                },
            ]
        }
    }
    plot_comparative_three_way_dsa(three_way, output_dir=str(fig_dir))
    assert any(fig_dir.iterdir())


def test_dsa_workflow_and_dashboard(tmp_path, base_params):
    figs = tmp_path / "dsa_workflow"
    figs.mkdir()
    interventions = {"Demo": base_params}

    one_way = perform_one_way_dsa(interventions, wtp_threshold=20000, n_points=2)
    plot_one_way_dsa_tornado(one_way, output_dir=str(figs))

    two_way = perform_comprehensive_two_way_dsa(
        interventions, wtp_threshold=20000, n_points=2
    )
    plot_two_way_dsa_heatmaps(two_way, output_dir=str(figs))

    three_way = perform_three_way_dsa(interventions, wtp_threshold=20000, n_points=2)
    plot_three_way_dsa_3d(three_way, output_dir=str(figs))

    compose_dsa_dashboard(output_dir=str(figs))
    assert any(figs.iterdir())


def test_additional_visuals_and_dashboards(tmp_path):
    fig_dir = tmp_path / "viz_misc"
    fig_dir.mkdir()

    plot_survival_gof(
        km_time=[0, 1, 2],
        km_survival=[1.0, 0.9, 0.8],
        fitted_models={
            "Weibull": {"time": [0, 1, 2], "survival": [1.0, 0.85, 0.7], "aic": 10.0}
        },
        output_dir=str(fig_dir),
    )
    plot_societal_drivers(
        "Demo", {"Productivity": 100, "Carer Time": 50}, output_dir=str(fig_dir)
    )
    plot_evp_curve(
        [10000, 20000],
        hs_nmb={"A": [1.0, 2.0], "B": [0.5, 1.5]},
        soc_nmb={"A": [1.2, 2.2], "B": [0.7, 1.3]},
        output_dir=str(fig_dir),
    )
    plot_structural_tornado(
        base_nmb=1000.0,
        scenario_ranges={"Discount": (800.0, 1200.0), "Perspective": (900.0, 1100.0)},
        output_dir=str(fig_dir),
    )
    plot_inequality_staircase(["Stage1", "Stage2"], [1.1, 0.9], output_dir=str(fig_dir))
    plot_financial_risk_protection(["A", "B"], [10, 20], output_dir=str(fig_dir))
    plot_inequality_aversion_sensitivity(
        pd.DataFrame(
            {
                "epsilon": [0.0, 0.5, 1.0],
                "atkinson_index": [0.1, 0.2, 0.3],
                "ede_net_benefit": [100, 90, 80],
                "mean_net_benefit": [110, 110, 110],
            }
        ),
        intervention_name="Strategy A",
        output_dir=str(fig_dir),
    )

    # Create dummy images to drive dashboard composition
    import matplotlib.pyplot as plt

    for name in [
        "annual_cash_flow_demo.png",
        "net_cash_flow_waterfall_demo.png",
        "affordability_ribbon_demo.png",
        "equity_impact_plane_demo.png",
        "lorenz_curve_demo.png",
        "equity_efficiency_plane_demo.png",
    ]:
        path = fig_dir / name
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(path)
        plt.close(fig)

    compose_bia_dashboard(output_dir=str(fig_dir))
    compose_equity_dashboard(["Demo"], output_dir=str(fig_dir))

    assert any(fig_dir.iterdir())
