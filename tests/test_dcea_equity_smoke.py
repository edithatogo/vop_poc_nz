import pandas as pd

from src.dcea_equity_analysis import (
    calculate_atkinson_index,
    calculate_gini,
    generate_dcea_results_table,
    plot_lorenz_curve,
)


def test_dcea_equity_smoke(tmp_path):
    out_dir = tmp_path / "equity_smoke"
    out_dir.mkdir()

    dcea_results = {
        "distribution_of_net_health_benefits": {"Q1": 100, "Q2": 120, "Q3": 140},
        "total_health_gain": 360,
        "variance_of_net_health_benefits": 200,
        "gini_coefficient": calculate_gini([100, 120, 140]),
        "atkinson_index": calculate_atkinson_index([100, 120, 140]),
        "atkinson_epsilon": 0.5,
    }
    table = generate_dcea_results_table(dcea_results, "Test")
    assert not table.empty

    plot_lorenz_curve(dcea_results, intervention_name="Test", output_dir=str(out_dir))
    # No equity_index_comparison helper exported; ensure Lorenz plot executes
    assert any(out_dir.iterdir())
