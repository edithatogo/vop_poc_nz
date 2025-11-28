import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.sobol_analysis import SobolAnalyzer, plot_sobol_indices


class TestSobolCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.param_distributions = {
            "p1": {"distribution": "uniform", "params": {"low": 0, "high": 1}},
            "p2": {"distribution": "normal", "params": {"mean": 0, "std": 1}},
            "p3": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}},
            "p4": {"distribution": "gamma", "params": {"shape": 1, "scale": 1}}
        }
        self.model_func = lambda params: params["p1"] + params["p2"]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        analyzer = SobolAnalyzer(self.model_func, self.param_distributions, n_samples=10)
        self.assertEqual(analyzer.n_samples, 10)
        self.assertEqual(len(analyzer.param_names), 4)

    def test_generate_sobol_sequence(self):
        analyzer = SobolAnalyzer(self.model_func, self.param_distributions, n_samples=10)
        seq = analyzer.generate_sobol_sequence(2, 10)
        self.assertEqual(seq.shape, (10, 2))
        self.assertTrue(np.all(seq >= 0) and np.all(seq <= 1))

    def test_sample_parameters(self):
        analyzer = SobolAnalyzer(self.model_func, self.param_distributions, n_samples=10)
        samples_01 = np.random.rand(10, 4)
        df = analyzer.sample_parameters(samples_01)
        self.assertEqual(df.shape, (10, 4))
        self.assertIn("p1", df.columns)
        self.assertIn("p2", df.columns)
        self.assertIn("p3", df.columns)
        self.assertIn("p4", df.columns)

    def test_saltelli_sampling(self):
        analyzer = SobolAnalyzer(self.model_func, self.param_distributions, n_samples=10)
        mat_A, mat_B, mats_AB = analyzer.saltelli_sampling()
        self.assertEqual(mat_A.shape, (10, 4))
        self.assertEqual(mat_B.shape, (10, 4))
        self.assertEqual(len(mats_AB), 4)
        self.assertEqual(mats_AB[0].shape, (10, 4))

    def test_evaluate_model(self):
        analyzer = SobolAnalyzer(self.model_func, self.param_distributions, n_samples=10)
        df = pd.DataFrame({
            "p1": [1, 2],
            "p2": [3, 4],
            "p3": [0, 0],
            "p4": [0, 0]
        })
        outputs = analyzer.evaluate_model(df)
        self.assertTrue(np.array_equal(outputs, np.array([4, 6])))

        # Test tuple output handling
        def tuple_model(params):
            return (100, 10) # cost, qaly -> NMB = 100 - 10*50000
        analyzer_tuple = SobolAnalyzer(tuple_model, self.param_distributions, n_samples=2)
        outputs_tuple = analyzer_tuple.evaluate_model(df)
        self.assertEqual(outputs_tuple[0], 100 - 10 * 50000)

        # Test failure handling
        def failing_model(params):
            raise ValueError("Fail")
        analyzer_fail = SobolAnalyzer(failing_model, self.param_distributions, n_samples=2)
        outputs_fail = analyzer_fail.evaluate_model(df)
        self.assertTrue(np.isnan(outputs_fail[0]))

    def test_calculate_sobol_indices(self):
        # Use a very simple model y = x1 + x2 where x1, x2 ~ U(0,1)
        # Variances should be roughly equal
        params = {
            "x1": {"distribution": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"distribution": "uniform", "params": {"low": 0, "high": 1}}
        }
        model = lambda p: p["x1"] + p["x2"]
        analyzer = SobolAnalyzer(model, params, n_samples=100)
        results = analyzer.calculate_sobol_indices()

        self.assertIn("indices", results)
        self.assertIn("confidence_intervals", results)
        indices = results["indices"]
        self.assertEqual(len(indices), 2)
        self.assertIn("first_order", indices.columns)
        self.assertIn("total_order", indices.columns)

    @patch("matplotlib.pyplot.subplots")
    def test_plot_sobol_indices(self, mock_subplots):
        # Setup mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        results = {
            "indices": pd.DataFrame({
                "parameter": ["p1", "p2"],
                "first_order": [0.4, 0.5],
                "total_order": [0.45, 0.55],
                "interaction": [0.05, 0.05]
            }),
            "confidence_intervals": pd.DataFrame(),
            "n_evaluations": 1000
        }
        plot_sobol_indices(results, output_dir=self.test_dir)

        # Verify savefig was called on the figure object
        self.assertTrue(mock_fig.savefig.called)

if __name__ == "__main__":
    unittest.main()
