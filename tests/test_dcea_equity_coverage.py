import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from vop_poc_nz.dcea_equity_analysis import (
    apply_equity_weights,
    calculate_atkinson_index,
    calculate_gini,
    calculate_inequality_aversion_sensitivity,
    generate_dcea_results_table,
    plot_combined_lorenz_curves,
    plot_comparative_equity_impact_plane,
    plot_equity_impact_plane,
    plot_lorenz_curve,
    plot_probabilistic_equity_impact_plane,
    plot_probabilistic_equity_impact_plane_with_delta,
    run_dcea,
)


class TestDCEAEquityCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_calculate_gini(self):
        # Test perfect equality
        self.assertAlmostEqual(calculate_gini([10, 10, 10]), 0.0)
        # Test perfect inequality (approximate)
        self.assertGreater(calculate_gini([0, 0, 10]), 0.6)
        # Test negative values handling
        self.assertGreaterEqual(calculate_gini([-10, 10, 20]), 0.0)

    def test_calculate_atkinson_index(self):
        # Test epsilon=1 special case
        self.assertGreaterEqual(calculate_atkinson_index([10, 20, 30], epsilon=1.0), 0.0)
        # Test epsilon!=1
        self.assertGreaterEqual(calculate_atkinson_index([10, 20, 30], epsilon=0.5), 0.0)

    def test_apply_equity_weights(self):
        nhb = {"A": 100, "B": 200}
        weights = {"A": 1.5, "B": 0.5}
        expected = 100 * 1.5 + 200 * 0.5
        self.assertEqual(apply_equity_weights(nhb, weights), expected)

    def test_run_dcea(self):
        subgroup_results = {
            "Group A": {"incremental_nmb": 100},
            "Group B": {"incremental_nmb": 200}
        }
        results = run_dcea(subgroup_results, epsilon=0.5, equity_weights={"Group A": 1.0, "Group B": 1.0})
        self.assertIn("gini_coefficient", results)
        self.assertIn("atkinson_index", results)
        self.assertEqual(results["total_health_gain"], 300)

    def test_generate_dcea_results_table(self):
        dcea_results = {
            "total_health_gain": 300,
            "weighted_total_health_gain": 300,
            "equity_weights": {"A": 1},
            "gini_coefficient": 0.2,
            "atkinson_index": 0.1,
            "variance_of_net_health_benefits": 50,
            "distribution_of_net_health_benefits": {"A": 100, "B": 200},
            "atkinson_epsilon": 0.5
        }
        df = generate_dcea_results_table(dcea_results, "Intervention X")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_calculate_inequality_aversion_sensitivity(self):
        subgroup_results = {
            "Group A": {"incremental_nmb": 100},
            "Group B": {"incremental_nmb": 200}
        }
        df = calculate_inequality_aversion_sensitivity(subgroup_results)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("epsilon", df.columns)
        self.assertIn("atkinson_index", df.columns)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_lorenz_curve(self, mock_savefig):
        dcea_results = {
            "societal": {
                "distribution_of_net_health_benefits": {"A": 100, "B": 200},
                "total_health_gain": 300
            }
        }
        plot_lorenz_curve(dcea_results, "Intervention X", output_dir=self.test_dir)
        # Check if file creation was attempted (mocked)
        self.assertTrue(mock_savefig.called)

        # Test empty/no societal
        plot_lorenz_curve({}, "Intervention Y", output_dir=self.test_dir)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_equity_impact_plane(self, mock_savefig):
        dcea_results_dict = {
            "health_system": {"total_health_gain": 100, "weighted_total_health_gain": 110},
            "societal": {"total_health_gain": 120, "weighted_total_health_gain": 130}
        }
        plot_equity_impact_plane(dcea_results_dict, "Intervention X", output_dir=self.test_dir)
        self.assertTrue(mock_savefig.called)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_comparative_equity_impact_plane(self, mock_savefig):
        all_dcea_results = {
            "Int A": {
                "health_system": {"total_health_gain": 100},
                "societal": {"total_health_gain": 120}
            },
            "Int B": {
                "health_system": {"total_health_gain": 200},
                "societal": {"total_health_gain": 220}
            }
        }
        plot_comparative_equity_impact_plane(all_dcea_results, output_dir=self.test_dir)
        self.assertTrue(mock_savefig.called)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_probabilistic_equity_impact_plane(self, mock_savefig):
        prob_results = {
            "Int A": pd.DataFrame({
                "inc_nmb_hs": [100, 110],
                "equity_weighted_nmb_hs": [105, 115],
                "inc_nmb_soc": [120, 130],
                "equity_weighted_nmb_soc": [125, 135]
            })
        }
        plot_probabilistic_equity_impact_plane(prob_results, output_dir=self.test_dir)
        self.assertTrue(mock_savefig.called)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_probabilistic_equity_impact_plane_with_delta(self, mock_savefig):
        prob_results = {
            "Int A": pd.DataFrame({
                "inc_nmb_hs": [100, 110],
                "equity_weighted_nmb_hs": [105, 115],
                "inc_nmb_soc": [120, 130],
                "equity_weighted_nmb_soc": [125, 135]
            })
        }
        plot_probabilistic_equity_impact_plane_with_delta(prob_results, output_dir=self.test_dir)
        self.assertTrue(mock_savefig.called)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_combined_lorenz_curves(self, mock_savefig):
        dcea_results_all = {
            "Int A": {
                "health_system": {
                    "distribution_of_net_health_benefits": {"G1": 10, "G2": 20}
                },
                "societal": {
                    "distribution_of_net_health_benefits": {"G1": 15, "G2": 25}
                }
            }
        }
        plot_combined_lorenz_curves(dcea_results_all, output_dir=self.test_dir)
        self.assertTrue(mock_savefig.called)

if __name__ == "__main__":
    unittest.main()
