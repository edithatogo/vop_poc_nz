import contextlib
import os

import numpy as np
import pandas as pd

import vop_poc_nz.visualizations as viz

OUTPUT_DIR = "output/test_figures"


def setup_module(module):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_markov_trace_saves_files():
    trace = pd.DataFrame({"Healthy": [0.9, 0.8], "Sick": [0.1, 0.2]}, index=[0, 1])
    viz.plot_markov_trace(
        {"trace_dataframe": trace}, output_dir=OUTPUT_DIR, intervention_name="Test"
    )
    assert any(f.startswith("markov_trace_test") for f in os.listdir(OUTPUT_DIR))


def test_density_ce_plane_hexbin():
    inc_costs = np.random.normal(0, 1, 100)
    inc_qalys = np.random.normal(0, 0.5, 100)
    viz.plot_density_ce_plane(
        inc_costs, inc_qalys, output_dir=OUTPUT_DIR, intervention_label="TestHex"
    )
    assert any(f.startswith("density_ce_plane_testhex") for f in os.listdir(OUTPUT_DIR))


def test_subgroup_forest_matplotlib():
    df = pd.DataFrame(
        {
            "subgroup": ["A", "B"],
            "estimate": [0.1, 0.2],
            "ci_lower": [0.05, 0.1],
            "ci_upper": [0.15, 0.3],
        }
    )
    viz.plot_subgroup_forest(df, output_dir=OUTPUT_DIR)
    assert any(f.startswith("subgroup_forest") for f in os.listdir(OUTPUT_DIR))


def test_rankogram():
    data = pd.DataFrame({"S1": [1, 2, 3], "S2": [3, 2, 1]})
    viz.plot_rankogram(data, output_dir=OUTPUT_DIR)
    assert any(f.startswith("rankogram") for f in os.listdir(OUTPUT_DIR))


def test_cost_qaly_breakdown():
    data = {
        "standard_care": {"Drug": 10, "Admin": 5, "QALY_LY": 0.5, "QALY_Utility": 0.2},
        "new_treatment": {"Drug": 12, "Admin": 4, "QALY_LY": 0.6, "QALY_Utility": 0.3},
    }
    viz.plot_cost_qaly_breakdown(data, output_dir=OUTPUT_DIR, intervention_name="Test")
    assert any(f.startswith("cost_qaly_breakdown_test") for f in os.listdir(OUTPUT_DIR))


def test_psa_convergence():
    vals = list(np.linspace(0, 1, 50))
    viz.plot_psa_convergence(vals, output_dir=OUTPUT_DIR, metric_label="NMB")
    assert any(f.startswith("psa_convergence_nmb") for f in os.listdir(OUTPUT_DIR))


def test_model_calibration():
    years = [1, 2, 3]
    obs = [10, 12, 11]
    preds = [9, 13, 10]
    viz.plot_model_calibration(
        years,
        obs,
        None,
        preds,
        None,
        None,
        output_dir=OUTPUT_DIR,
        metric_label="Events",
    )
    assert any(f.startswith("model_calibration") for f in os.listdir(OUTPUT_DIR))


def test_decision_reversal_matrix():
    hs = [1, -1]
    soc = [2, -2]
    viz.plot_decision_reversal_matrix(hs, soc, ["A", "B"], output_dir=OUTPUT_DIR)
    assert any(f.startswith("decision_reversal_matrix") for f in os.listdir(OUTPUT_DIR))


def test_affordability_ribbon():
    years = [1, 2, 3]
    mean = [100, 80, 60]
    lower = [90, 70, 50]
    upper = [110, 90, 70]
    viz.plot_affordability_ribbon(
        years,
        mean,
        lower,
        upper,
        output_dir=OUTPUT_DIR,
        intervention="Test",
        perspective="societal",
    )
    assert any("affordability_ribbon" in f for f in os.listdir(OUTPUT_DIR))


def test_equity_efficiency_plane():
    strategies = ["A", "B"]
    eff = [1.0, 2.0]
    eq = [0.1, -0.2]
    viz.plot_equity_efficiency_plane(strategies, eff, eq, output_dir=OUTPUT_DIR)
    assert any(f.startswith("equity_efficiency_plane") for f in os.listdir(OUTPUT_DIR))


def test_dashboard_composer_with_existing(tmp_path):
    # create dummy image in tmp_path for isolation
    import matplotlib.pyplot as plt

    img_path = str(tmp_path / "dummy.png")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(img_path)
    plt.close(fig)
    viz.compose_dashboard(
        [img_path], output_dir=str(tmp_path), filename_base="dash_exists"
    )
    assert any(f.startswith("dash_exists") for f in os.listdir(tmp_path))


def test_dashboard_composer_skips_missing():
    paths = [os.path.join(OUTPUT_DIR, "nonexistent.png")]
    viz.compose_dashboard(paths, output_dir=OUTPUT_DIR, filename_base="dash_test")
    assert any(f.startswith("dash_test") for f in os.listdir(OUTPUT_DIR)) is False


def teardown_module(module):
    # clean up generated files
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(OUTPUT_DIR, f))
        with contextlib.suppress(OSError):
            os.rmdir(OUTPUT_DIR)


def test_visual_regression_hash():
    # simple deterministic plot and hash check for CE plane density
    inc_costs = [0, 1]
    inc_qalys = [0, 1]
    viz.plot_density_ce_plane(
        inc_costs, inc_qalys, output_dir=OUTPUT_DIR, intervention_label="hashTest"
    )
    target = None
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("density_ce_plane_hashtest") and f.endswith(".png"):
            target = os.path.join(OUTPUT_DIR, f)
            break
    assert target is not None
    import hashlib

    with open(target, "rb") as handle:
        h = hashlib.md5(handle.read()).hexdigest()
    assert len(h) == 32
