import numpy as np
import pandas as pd

from src.visualizations import (
    plot_cumulative_nmb,
    plot_efficiency_frontier,
    plot_equity_efficiency_plane,
    plot_expected_loss_curve,
    plot_price_acceptability_curve,
    plot_psa_convergence,
    plot_rankogram,
    plot_resource_constraint,
    plot_scenario_waterfall,
    plot_threshold_crossing,
)


def test_additional_plots(tmp_path):
    out_dir = tmp_path / "viz_extra"
    out_dir.mkdir()

    # Efficiency frontier
    plot_efficiency_frontier(
        costs=[10000, 12000, 15000],
        qalys=[10.0, 11.0, 12.5],
        strategy_names=["A", "B", "C"],
        output_dir=str(out_dir),
    )

    # Equity-efficiency plane
    plot_equity_efficiency_plane(
        strategies=["A", "B"],
        efficiency_gains=[0.5, 1.0],
        equity_gains=[0.1, -0.2],
        output_dir=str(out_dir),
    )

    # Rankogram
    psa_nmb = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 1], "C": [3, 3, 2]})
    plot_rankogram(psa_nmb, output_dir=str(out_dir))

    # PSA convergence
    plot_psa_convergence(values=list(np.linspace(0, 1, 50)), output_dir=str(out_dir))

    # Threshold crossing
    plot_threshold_crossing(
        param_range=list(np.linspace(0, 1, 20)),
        nmb_values=list(np.linspace(-100, 100, 20)),
        param_name="Test Param",
        output_dir=str(out_dir),
    )

    # Scenario waterfall
    plot_scenario_waterfall(
        base_value=100.0,
        scenarios={"High uptake": 120.0, "Low uptake": 80.0},
        output_dir=str(out_dir),
    )

    # Expected loss curve requires mapping WTP -> NMB DataFrame
    psa_nmb_by_wtp = {
        10000: pd.DataFrame({"A": [1, 2], "B": [2, 1]}),
        20000: pd.DataFrame({"A": [2, 3], "B": [3, 2]}),
    }
    plot_expected_loss_curve(psa_nmb_by_wtp, output_dir=str(out_dir))

    # Price acceptability
    plot_price_acceptability_curve(
        wtp_range=[10000, 20000, 30000],
        break_even_prices=[100, 200, 300],
        output_dir=str(out_dir),
    )

    # Resource constraint
    plot_resource_constraint(
        annual_resource_use=[100, 120, 140],
        capacity_limit=130,
        output_dir=str(out_dir),
    )

    # Cumulative NMB
    plot_cumulative_nmb([100, -50, 80], output_dir=str(out_dir))

    assert any(out_dir.iterdir())
