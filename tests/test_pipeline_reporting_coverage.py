import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.pipeline.reporting import run_reporting_pipeline


class TestPipelineReportingCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.results = {
            "reports": {"Intervention_A": "Report Content"},
            "selected_interventions": {
                "Intervention_A": {
                    "states": ["Healthy", "Sick"],
                    "bia_population": {"total_population": 1000}
                }
            },
            "intervention_results": {
                "Intervention_A": {
                    "health_system": {
                        "trace_standard_care": np.array([[1.0, 0.0]]),
                        "trace_new_treatment": np.array([[0.9, 0.1]])
                    },
                    "societal": {
                        "human_capital": {
                            "inequality_sensitivity": pd.DataFrame({"epsilon": [0.5], "atkinson": [0.1]})
                        }
                    },
                    "discordance": {"metric": 100}
                }
            },
            "probabilistic_results": {
                "Intervention_A": pd.DataFrame({"inc_nmb": [100, 200]})
            },
            "dsa_analysis": {
                "1_way": {"Intervention_A": {}},
                "2_way": {},
                "3_way": {}
            },
            "bia_results": {
                "Intervention_A": pd.DataFrame({
                    "year": [1, 2],
                    "gross_cost": [100, 100],
                    "net_cost": [80, 80]
                })
            },
            "voi_analysis": {
                "Intervention_A": {}
            },
            "dcea_equity_analysis": {
                "Intervention_A": {"societal": {}}
            }
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.pipeline.reporting.generate_policy_brief")
    @patch("src.pipeline.reporting.compose_dashboard")
    @patch("src.pipeline.reporting.plot_discordance_loss")
    @patch("src.pipeline.reporting.plot_inequality_aversion_sensitivity")
    @patch("src.pipeline.reporting.plot_combined_lorenz_curves")
    @patch("src.pipeline.reporting.plot_probabilistic_equity_impact_plane_with_delta")
    @patch("src.pipeline.reporting.plot_probabilistic_equity_impact_plane")
    @patch("src.pipeline.reporting.plot_comparative_equity_impact_plane")
    @patch("src.pipeline.reporting.plot_markov_trace")
    @patch("src.pipeline.reporting.plot_comparative_bia_line")
    @patch("src.pipeline.reporting.plot_annual_cash_flow")
    @patch("src.pipeline.reporting.compose_dsa_dashboard")
    @patch("src.pipeline.reporting.plot_three_way_dsa_3d")
    @patch("src.pipeline.reporting.plot_two_way_dsa_heatmaps")
    @patch("src.pipeline.reporting.plot_one_way_dsa_tornado")
    @patch("src.pipeline.reporting.plot_decision_tree")
    @patch("src.pipeline.reporting.plot_comparative_evppi_with_delta")
    @patch("src.pipeline.reporting.plot_comparative_evpi_with_delta")
    @patch("src.pipeline.reporting.plot_comparative_ceac_with_delta")
    @patch("src.pipeline.reporting.plot_comparative_ce_plane_with_delta")
    @patch("src.pipeline.reporting.plot_comparative_pop_evpi_with_delta")
    @patch("src.pipeline.reporting.plot_comparative_evppi")
    @patch("src.pipeline.reporting.plot_pop_evpi")
    @patch("src.pipeline.reporting.plot_value_of_perspective")
    @patch("src.pipeline.reporting.plot_net_benefit_curves")
    @patch("src.pipeline.reporting.plot_comparative_evpi")
    @patch("src.pipeline.reporting.plot_ceaf")
    @patch("src.pipeline.reporting.plot_comparative_ceac")
    @patch("src.pipeline.reporting.plot_ceac")
    @patch("src.pipeline.reporting.plot_comparative_ce_plane")
    @patch("src.pipeline.reporting.plot_cost_effectiveness_plane")
    @patch("src.pipeline.reporting.generate_all_tables")
    def test_run_reporting_pipeline(self, *mocks):
        # Run the pipeline
        run_reporting_pipeline(self.results, self.test_dir)

        # Verify JSON dump
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "complete_analysis_results.json")))

        # Verify Report creation
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "combined_report.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "Intervention_A_report.md")))

if __name__ == "__main__":
    unittest.main()
