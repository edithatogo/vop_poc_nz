"""
Unit tests for the canonical health economic analysis code.

These tests verify the corrections made and improvements implemented
to address reviewer feedback.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cea_model_core import (
    MarkovModel,
    _calculate_cer,
    _calculate_icer,
    create_parameters_table,
    run_cea,
)
from src.dcea_analysis import DCEDataProcessor
from src.value_of_information import (
    ProbabilisticSensitivityAnalysis,
    calculate_evpi,
)


class TestCorrectedCEACalculations(unittest.TestCase):
    """Test the corrected CEA calculations."""

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
        }

    def test_markov_model_validation(self):
        """Test that Markov model validates inputs properly."""
        # Valid parameters should work
        model_instance = MarkovModel(
            states=["A", "B"], transition_matrix=[[0.8, 0.2], [0.1, 0.9]]
        )
        assert isinstance(model_instance, MarkovModel)

        # Invalid transition matrix should raise error (not summing to 1)
        with self.assertRaises(ValueError):
            MarkovModel(
                states=["A", "B"],
                transition_matrix=[[0.8, 0.3], [0.1, 0.9]],  # Row sums to 1.1
            )

    def test_icer_calculation_edge_cases(self):
        """Test ICER calculation with edge cases."""
        # Zero incremental QALYs with positive cost should give +inf
        icer = _calculate_icer(inc_cost=1000, inc_qalys=0)
        self.assertEqual(icer, float("inf"))

        # Zero incremental QALYs with negative cost should give -inf
        icer = _calculate_icer(inc_cost=-1000, inc_qalys=0)
        self.assertEqual(icer, float("-inf"))

        # Both zero should give 0
        icer = _calculate_icer(inc_cost=0, inc_qalys=0)
        self.assertEqual(icer, 0.0)

        # Normal case
        icer = _calculate_icer(inc_cost=10000, inc_qalys=2)
        self.assertEqual(icer, 5000)

    def test_cer_calculation(self):
        """Test cost-effectiveness ratio calculation."""
        # Zero QALYs with positive cost should give +inf
        cer = _calculate_cer(cost=1000, qalys=0)
        self.assertEqual(cer, float("inf"))

        # Normal case
        cer = _calculate_cer(cost=10000, qalys=2)
        self.assertEqual(cer, 5000)

    def test_run_cea_basic(self):
        """Test basic CEA run functionality."""
        results = run_cea(self.test_params, perspective="health_system")

        # Check that results contain expected keys
        expected_keys = [
            "perspective",
            "cost_standard_care",
            "qalys_standard_care",
            "cost_new_treatment",
            "qalys_new_treatment",
            "incremental_cost",
            "incremental_qalys",
            "icer",
            "incremental_nmb",
        ]

        for key in expected_keys:
            self.assertIn(key, results)

        # Check that perspective is correctly set
        self.assertEqual(results["perspective"], "health_system")

    def test_perspective_difference(self):
        """Test that health system and societal perspectives give different results."""
        hs_results = run_cea(self.test_params, perspective="health_system")
        s_results = run_cea(self.test_params, perspective="societal")

        # Costs should be different between perspectives
        self.assertNotEqual(
            hs_results["cost_new_treatment"], s_results["cost_new_treatment"]
        )

        # ICERs should be different
        self.assertNotEqual(hs_results["icer"], s_results["icer"])


class TestDCEAImplementation(unittest.TestCase):
    """Test the DCEA implementation."""

    def test_dce_data_processor(self):
        """Test basic DCE data processor functionality."""
        processor = DCEDataProcessor()

        # Define simple attributes for testing
        attributes = {
            "perspective": {
                "levels": ["health_system", "societal"],
                "type": "categorical",
                "description": "Evaluation perspective",
            },
            "cost_per_qaly": {
                "levels": [30000, 50000],
                "type": "continuous",
                "description": "Cost per QALY threshold",
            },
        }

        processor.define_attributes(attributes)

        # Check that attributes are stored correctly
        self.assertEqual(processor.attribute_definitions, attributes)


class TestValueOfInformation(unittest.TestCase):
    """Test the value of information implementations."""

    def test_evpi_basic(self):
        """Test basic EVPI calculation."""
        # Create dummy PSA results DataFrame
        np.random.seed(42)  # For reproducible tests
        n = 100

        psa_results = pd.DataFrame(
            {
                "cost_sc": np.random.normal(10000, 1000, n),
                "qaly_sc": np.random.normal(5.0, 0.5, n),
                "cost_nt": np.random.normal(15000, 1500, n),
                "qaly_nt": np.random.normal(6.0, 0.6, n),
            }
        )

        # EVPI should be non-negative
        evpi = calculate_evpi(psa_results, wtp_threshold=50000)
        self.assertGreaterEqual(evpi, 0)

    def test_probabilistic_sensitivity_analysis(self):
        """Test basic PSA functionality."""

        # Simple model function for testing
        def simple_model(params, intervention_type="standard_care"):
            base_cost = params.get("base_cost", 10000)
            base_qaly = params.get("base_qaly", 5.0)
            return float(base_cost), float(base_qaly)

        # Define simple parameter distributions
        params = {
            "base_cost": {
                "distribution": "gamma",
                "params": {"shape": 10, "scale": 1000},
            },
            "base_qaly": {"distribution": "beta", "params": {"alpha": 8, "beta": 2}},
        }

        psa = ProbabilisticSensitivityAnalysis(
            simple_model, params, wtp_threshold=50000
        )

        # This should run without errors
        try:
            psa_results = psa.run_psa(n_samples=10)  # Small sample for test speed
            self.assertIsInstance(psa_results, pd.DataFrame)
            self.assertEqual(len(psa_results), 10)
        except Exception as e:
            self.fail(f"PSA run failed with error: {e}")


class TestTransparencyFeatures(unittest.TestCase):
    """Test the transparency and documentation features."""

    def setUp(self):
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
        }

    def test_parameters_table_creation(self):
        """Test that parameters table is created correctly."""
        sources = {
            "cycles": "Model specification",
            "cost_hs_Sick_standard": "Literature reference",
        }

        params_table = create_parameters_table(self.test_params, sources)

        # Check that the table has the expected structure
        expected_columns = [
            "Parameter",
            "Value",
            "Description",
            "Source",
            "Perspective",
        ]
        for col in expected_columns:
            self.assertIn(col, params_table.columns)

        # Check that we have some parameters documented
        self.assertGreater(len(params_table), 0)
