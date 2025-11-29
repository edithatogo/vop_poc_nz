"""
Unit tests for the new features added to the canonical health economic analysis code.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from vop_poc_nz.bia_model import bia_to_markdown_table, project_bia
from vop_poc_nz.discordance_analysis import calculate_decision_discordance
from vop_poc_nz.dsa_analysis import (
    perform_comprehensive_two_way_dsa,
    perform_three_way_dsa,
)
from vop_poc_nz.reporting import generate_comprehensive_report
from vop_poc_nz.threshold_analysis import run_threshold_analysis


class TestNewFeatures(unittest.TestCase):
    """Test the new features added to the canonical codebase."""

    def setUp(self):
        """Set up test parameters."""
        self.test_params = {
            "states": ["Healthy", "Sick", "Dead"],
            "transition_matrices": {
                "standard_care": [[0.9, 0.05, 0.05], [0, 0.8, 0.2], [0, 0, 1]],
                "new_treatment": [[0.95, 0.03, 0.02], [0, 0.85, 0.15], [0, 0, 1]],
            },
            "cycles": 10,
            "initial_population": [1000, 0, 0],
            "costs": {
                "health_system": {
                    "standard_care": [0, 500, 0],
                    "new_treatment": [100, 500, 0],
                },
                "societal": {
                    "standard_care": [0, 1000, 0],
                    "new_treatment": [0, 500, 0],
                },
            },
            "qalys": {"standard_care": [1, 0.7, 0], "new_treatment": [1, 0.8, 0]},
            "productivity_costs": {
                "human_capital": {
                    "standard_care": [0, 1000, 0],
                    "new_treatment": [0, 500, 0],
                }
            },
            "discount_rate": 0.03,
        }
        self.interventions = {"Test Intervention": self.test_params}

    def test_dsa_analysis(self):
        """Test the DSA analysis functions."""
        dsa_results_2_way = perform_comprehensive_two_way_dsa(
            self.interventions, wtp_threshold=50000, n_points=5
        )
        self.assertIn("Test Intervention", dsa_results_2_way)
        self.assertIn("param1_name", dsa_results_2_way["Test Intervention"])
        self.assertIn("param2_name", dsa_results_2_way["Test Intervention"])
        self.assertIn("dsa_grid_hs", dsa_results_2_way["Test Intervention"])
        self.assertIn("dsa_grid_soc", dsa_results_2_way["Test Intervention"])

        dsa_results_3_way = perform_three_way_dsa(
            self.interventions, wtp_threshold=50000, n_points=5
        )
        self.assertIn("Test Intervention", dsa_results_3_way)
        self.assertIn("param1_name", dsa_results_3_way["Test Intervention"])
        self.assertIn("param2_name", dsa_results_3_way["Test Intervention"])
        self.assertIn("param3_name", dsa_results_3_way["Test Intervention"])
        self.assertIn("dsa_grid", dsa_results_3_way["Test Intervention"])

    def test_discordance_analysis(self):
        """Test the discordance analysis function."""
        discordance_results = calculate_decision_discordance(
            "Test Intervention", self.test_params
        )
        self.assertIn("discordant", discordance_results)
        self.assertIsInstance(discordance_results["discordant"], bool)
        self.assertIn("loss_from_discordance", discordance_results)
        self.assertIsInstance(discordance_results["loss_from_discordance"], float)

    def test_threshold_analysis(self):
        """Test the threshold analysis function."""
        parameter_ranges = {"new_treatment_cost_multiplier": np.linspace(0.5, 1.5, 5)}
        threshold_results = run_threshold_analysis(
            "Test Intervention", self.test_params, parameter_ranges
        )
        self.assertIn("new_treatment_cost_multiplier", threshold_results)
        self.assertIsInstance(
            threshold_results["new_treatment_cost_multiplier"], pd.DataFrame
        )
        self.assertGreater(len(threshold_results["new_treatment_cost_multiplier"]), 0)

    def test_bia_model(self):
        """Test the BIA model functions."""
        bia_params = {
            "population_size": 100000,
            "eligible_prop": 0.1,
            "uptake_by_year": [0.1, 0.2, 0.3, 0.4, 0.5],
            "cost_per_patient": self.test_params["costs"]["health_system"][
                "new_treatment"
            ][0],
            "offset_cost_per_patient": self.test_params["costs"]["health_system"][
                "standard_care"
            ][0],
        }
        bia_df = project_bia(**bia_params)
        self.assertIsInstance(bia_df, pd.DataFrame)
        self.assertEqual(len(bia_df), 5)

        md_table = bia_to_markdown_table(bia_df)
        self.assertIsInstance(md_table, str)
        self.assertIn("Gross cost", md_table)

    def test_reporting(self):
        """Test the reporting function."""
        report = generate_comprehensive_report("Test Intervention", self.test_params)
        self.assertIsInstance(report, str)
        self.assertIn("Comprehensive CEA Report: Test Intervention", report)
        self.assertIn("Health System Perspective", report)
        self.assertIn("Societal Perspective", report)
        self.assertIn("Decision Discordance", report)


if __name__ == "__main__":
    unittest.main()
