import pandas as pd

from src.visualizations import (
    plot_ceac,
    plot_ceaf,
    plot_cost_effectiveness_plane,
    plot_evpi,
    plot_evppi,
    plot_net_benefit_curves,
    plot_pop_evpi,
    plot_value_of_perspective,
)


def test_ce_and_voi_plots(tmp_path):
    out_dir = tmp_path / "ce_figs"
    out_dir.mkdir()

    psa_df = pd.DataFrame(
        {
            "inc_cost": [100.0, 150.0, 120.0],
            "inc_qaly": [0.1, 0.15, 0.12],
            "cost_sc": [1000.0, 1100.0, 1050.0],
            "qaly_sc": [5.0, 5.1, 5.05],
            "cost_nt": [1200.0, 1250.0, 1230.0],
            "qaly_nt": [5.5, 5.4, 5.45],
        }
    )
    all_results = {"ModelA": psa_df}
    wtp_thresholds = [10000, 20000]

    plot_cost_effectiveness_plane(all_results, output_dir=str(out_dir))
    plot_ceac(all_results, wtp_thresholds, output_dir=str(out_dir))
    plot_ceaf(all_results, wtp_thresholds, output_dir=str(out_dir))
    plot_evpi(all_results, wtp_thresholds, output_dir=str(out_dir))
    plot_net_benefit_curves(all_results, wtp_thresholds, output_dir=str(out_dir))
    plot_value_of_perspective(all_results, wtp_thresholds, output_dir=str(out_dir))
    plot_pop_evpi(all_results, wtp_thresholds, population_sizes={"ModelA": 1000}, output_dir=str(out_dir))

    voi_report = {
        "value_of_information": {
            "evppi_by_parameter_group": {"EVPPI_demo": [1.0, 2.0]},
            "wtp_thresholds": wtp_thresholds,
        }
    }
    plot_evppi(voi_report, output_dir=str(out_dir))

    assert any(out_dir.iterdir())
