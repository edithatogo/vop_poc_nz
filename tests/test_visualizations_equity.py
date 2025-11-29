import pandas as pd

from vop_poc_nz.dcea_equity_analysis import plot_equity_impact_plane
from vop_poc_nz.visualizations import (
    plot_equity_efficiency_plane,
    plot_inequality_aversion_sensitivity,
    plot_inequality_staircase,
    plot_price_acceptability_curve,
    plot_societal_drivers,
)


def test_equity_plots(tmp_path):
    out_dir = tmp_path / "equity"
    out_dir.mkdir()

    # Equity-efficiency impact
    plot_equity_efficiency_plane(
        strategies=["A", "B"],
        efficiency_gains=[0.5, 1.0],
        equity_gains=[0.1, -0.2],
        output_dir=str(out_dir),
    )

    # Equity impact plane
    # Inequality aversion curve
    plot_inequality_aversion_sensitivity(
        pd.DataFrame(
            {
                "epsilon": [0.0, 0.5, 1.0],
                "atkinson_index": [0.1, 0.2, 0.3],
                "ede_net_benefit": [100, 90, 80],
                "mean_net_benefit": [110, 110, 110],
            }
        ),
        intervention_name="A",
        output_dir=str(out_dir),
    )

    # Inequality staircase
    stages = ["Eligibility", "Access", "Uptake", "Adherence", "Efficacy"]
    relative_risks = [1.0, 1.1, 1.2, 1.3, 1.1]
    plot_inequality_staircase(stages, relative_risks, output_dir=str(out_dir))

    # Societal drivers decomposition
    plot_societal_drivers(
        "Demo", {"Productivity": 1000, "Care": 500}, output_dir=str(out_dir)
    )

    # Equity impact plane using simple DCEA results
    dcea_results = {
        "total_health_gain": 100.0,
        "variance_of_net_health_benefits": 10.0,
        "distribution_of_net_health_benefits": {"Q1": 30, "Q2": 70},
    }
    plot_equity_efficiency_plane(
        ["A", "B"], [1.0, 2.0], [0.2, -0.1], output_dir=str(out_dir)
    )
    plot_equity_impact_plane(dcea_results, "Demo", output_dir=str(out_dir))

    # Price acceptability
    plot_price_acceptability_curve(
        wtp_range=[10000, 20000, 30000],
        break_even_prices=[100, 200, 300],
        output_dir=str(out_dir),
    )

    assert any(out_dir.iterdir())
