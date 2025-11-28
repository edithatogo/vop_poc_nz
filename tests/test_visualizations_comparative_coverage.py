import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import shutil
import tempfile
import os
from src.visualizations_comparative import (
    plot_comparative_cash_flow,
    plot_icer_ladder,
    plot_nmb_comparison,
    plot_equity_impact_comparison,
    plot_comprehensive_intervention_summary
)

class TestVisualizationsComparativeCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.bia_results = {
            "Int_A": pd.DataFrame({
                "year": [1, 2],
                "net_cost": [100, 200],
                "discounted_net_cost": [90, 180]
            }),
            "Int_B": pd.DataFrame({
                "year": [1, 2],
                "net_cost": [150, 250]
            })
        }
        self.intervention_results = {
            "Int_A": {
                "societal": {
                    "human_capital": {
                        "incremental_cost": 1000,
                        "incremental_qalys": 1.0,
                        "icer": 1000,
                        "incremental_nmb": 49000,
                        "dcea_equity_analysis": {
                            "atkinson_index": 0.1,
                            "gini_coefficient": 0.2,
                            "weighted_total_health_gain": 50000
                        }
                    }
                },
                "health_system": {
                    "incremental_cost": 1200,
                    "incremental_qalys": 1.0,
                    "icer": 1200,
                    "incremental_nmb": 48800
                }
            },
            "Int_B": {
                "societal": {
                    "incremental_cost": 500,
                    "incremental_qalys": 0.8,
                    "icer": 625,
                    "incremental_nmb": 39500
                },
                "health_system": {
                    "incremental_cost": 600,
                    "incremental_qalys": 0.8,
                    "icer": 750,
                    "incremental_nmb": 39400
                }
            }
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.visualizations.save_figure")
    @patch("src.visualizations_comparative.plt")
    @patch("src.visualizations.apply_default_style")
    def test_plot_comparative_cash_flow(self, mock_style, mock_plt, mock_save):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        plot_comparative_cash_flow(self.bia_results, self.test_dir, discount=True)
        self.assertTrue(mock_save.called)
        
        # Test nominal
        plot_comparative_cash_flow(self.bia_results, self.test_dir, discount=False)

    @patch("src.visualizations.save_figure")
    @patch("src.visualizations_comparative.plt")
    @patch("src.visualizations.apply_default_style")
    def test_plot_icer_ladder(self, mock_style, mock_plt, mock_save):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        plot_icer_ladder(self.intervention_results, output_dir=self.test_dir, perspective="societal")
        self.assertTrue(mock_save.called)
        
        # Test fallback to top-level results (Int_B structure)
        plot_icer_ladder(self.intervention_results, output_dir=self.test_dir, perspective="health_system")

    @patch("src.visualizations.save_figure")
    @patch("src.visualizations_comparative.plt")
    @patch("src.visualizations.apply_default_style")
    def test_plot_nmb_comparison(self, mock_style, mock_plt, mock_save):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        plot_nmb_comparison(self.intervention_results, output_dir=self.test_dir)
        self.assertTrue(mock_save.called)

    @patch("src.visualizations.save_figure")
    @patch("src.visualizations_comparative.plt")
    @patch("src.visualizations.apply_default_style")
    def test_plot_equity_impact_comparison(self, mock_style, mock_plt, mock_save):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax, mock_ax))
        
        plot_equity_impact_comparison(self.intervention_results, output_dir=self.test_dir)
        self.assertTrue(mock_save.called)
        
        # Test no equity data
        plot_equity_impact_comparison({"Int_C": {}}, output_dir=self.test_dir)

    @patch("src.visualizations.save_figure")
    @patch("src.visualizations_comparative.plt")
    @patch("src.visualizations.apply_default_style")
    def test_plot_comprehensive_intervention_summary(self, mock_style, mock_plt, mock_save):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        plot_comprehensive_intervention_summary(
            self.intervention_results,
            self.bia_results,
            output_dir=self.test_dir
        )
        self.assertTrue(mock_save.called)

if __name__ == "__main__":
    unittest.main()
