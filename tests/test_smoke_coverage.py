import pandas as pd

from src.main_analysis import generate_policy_implications_report
from src.value_of_information import generate_voi_report
from src.visualizations import (
    plot_cost_qaly_breakdown,
    plot_density_ce_plane,
    plot_markov_trace,
    plot_net_cash_flow_waterfall,
)


def test_visualization_smoke(tmp_path):
    out_dir = tmp_path / "figs"
    out_dir.mkdir()

    trace = pd.DataFrame({"Healthy": [0.9, 0.8], "Sick": [0.1, 0.2]}, index=[0, 1])
    plot_markov_trace(
        {"trace_dataframe": trace},
        output_dir=str(out_dir),
        intervention_name="smoke",
    )

    plot_density_ce_plane(
        inc_costs=[0.0, 1.0],
        inc_qalys=[0.0, 0.5],
        output_dir=str(out_dir),
        intervention_label="smoke",
    )

    cost_qaly_data = {
        "standard_care": {"Drug": 100, "QALY_LY": 0.4, "QALY_Utility": 0.1},
        "new_treatment": {"Drug": 120, "QALY_LY": 0.5, "QALY_Utility": 0.2},
    }
    plot_cost_qaly_breakdown(
        cost_qaly_data, output_dir=str(out_dir), intervention_name="smoke"
    )

    plot_net_cash_flow_waterfall(
        years=[1, 2, 3],
        investment_costs=[1000, 800, 500],
        cost_offsets=[200, 400, 700],
        output_dir=str(out_dir),
        intervention="smoke",
        perspective="test",
    )

    # Ensure at least one file was written
    assert any(out_dir.iterdir())


def test_reporting_and_voi_smoke(tmp_path):
    # Minimal intervention results to exercise policy implications path
    intervention_results = {
        "TestIntervention": {
            "health_system": {"icer": 10000.0, "incremental_nmb": 5000.0},
            "societal": {
                "human_capital": {"icer": 8000.0, "incremental_nmb": 7000.0}
            },
        }
    }
    report = generate_policy_implications_report(intervention_results)
    assert "intervention_level_differences" in report

    # VOI smoke run
    psa_df = pd.DataFrame(
        {
            "qaly_sc": [1.0, 0.9],
            "qaly_nt": [1.1, 1.0],
            "cost_sc": [1000, 900],
            "cost_nt": [1100, 950],
            "inc_cost": [100, 50],
            "inc_qaly": [0.1, 0.1],
        }
    )
    voi = generate_voi_report(psa_df, wtp_thresholds=[50000], target_population=1000)
    assert "value_of_information" in voi
    out_dir = tmp_path / "voi"
    out_dir.mkdir()
