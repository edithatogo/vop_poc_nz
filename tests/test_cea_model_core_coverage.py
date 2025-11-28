import unittest

import numpy as np
import pandas as pd

from src.cea_model_core import (
    MarkovModel,
    _calculate_cer,
    _calculate_friction_cost,
    _calculate_icer,
    _get_costs_qalys_by_perspective,
    _validate_model_parameters,
    create_parameters_table,
    deep_update,
    generate_comparative_icer_table,
    run_cea,
)


class TestCEAModelCore(unittest.TestCase):
    def setUp(self):
        self.states = ["Healthy", "Sick", "Dead"]
        self.transition_matrix = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.7, 0.2],
            [0.0, 0.0, 1.0]
        ])
        self.initial_pop = np.array([1000, 0, 0])
        self.costs = np.array([100, 500, 0])
        self.qalys = np.array([1.0, 0.7, 0.0])

        self.model_params = {
            "states": self.states,
            "transition_matrices": {
                "standard_care": self.transition_matrix.tolist(),
                "new_treatment": self.transition_matrix.tolist()
            },
            "initial_population": self.initial_pop.tolist(),
            "costs": {
                "health_system": {
                    "standard_care": self.costs.tolist(),
                    "new_treatment": self.costs.tolist()
                },
                "societal": {
                    "standard_care": (self.costs + 50).tolist(),
                    "new_treatment": (self.costs + 50).tolist()
                }
            },
            "qalys": {
                "standard_care": self.qalys.tolist(),
                "new_treatment": self.qalys.tolist()
            },
            "cycles": 10,
            "discount_rate": 0.03,
            "productivity_costs": {
                "human_capital": {
                    "standard_care": [10, 20, 0],
                    "new_treatment": [10, 20, 0]
                },
                "friction_cost": {
                    "standard_care": [5, 10, 0],
                    "new_treatment": [5, 10, 0]
                }
            }
        }

    def test_markov_model_init(self):
        model = MarkovModel(self.states, self.transition_matrix)
        self.assertEqual(len(model.states), 3)
        self.assertEqual(model.discount_rate, 0.03)

    def test_markov_model_run(self):
        model = MarkovModel(self.states, self.transition_matrix)
        total_cost, total_qalys, trace = model.run(10, self.initial_pop, self.costs, self.qalys)
        self.assertGreater(total_cost, 0)
        self.assertGreater(total_qalys, 0)
        self.assertEqual(trace.shape[0], 11)  # 10 cycles + initial

    def test_deep_update(self):
        d = {"a": {"b": 1}, "c": 2}
        u = {"a": {"d": 3}, "e": 4}
        result = deep_update(d, u)
        self.assertEqual(result["a"]["b"], 1)
        self.assertEqual(result["a"]["d"], 3)
        self.assertEqual(result["e"], 4)

    def test_run_cea(self):
        results = run_cea(self.model_params, perspective="health_system")
        self.assertIn("cost_standard_care", results)
        self.assertIn("cost_new_treatment", results)
        self.assertIn("icer", results)
        self.assertIn("incremental_nmb", results)

    def test_run_cea_societal(self):
        results = run_cea(self.model_params, perspective="societal", productivity_cost_method="human_capital")
        self.assertIn("cost_standard_care", results)
        self.assertIsInstance(results, dict)

    def test_run_cea_with_subgroups(self):
        params_with_subgroups = self.model_params.copy()
        params_with_subgroups["subgroups"] = {
            "Maori": {"initial_population": [500, 0, 0]},
            "Non-Maori": {"initial_population": [500, 0, 0]}
        }
        results = run_cea(params_with_subgroups, perspective="health_system")
        # Subgroup results should be returned but not as dcea_equity_analysis
        self.assertIsInstance(results, dict)

    def test_calculate_icer(self):
        # Normal case
        icer = _calculate_icer(1000, 1.0)
        self.assertEqual(icer, 1000)

        # Dominated (negative QALY)
        icer = _calculate_icer(1000, -0.5)
        self.assertTrue(icer < 0)  # Negative ICER

        # Dominant (negative cost, positive QALY)
        icer = _calculate_icer(-1000, 0.5)
        self.assertEqual(icer, -2000)  # Negative ICER

        # Zero QALY, positive cost
        icer = _calculate_icer(1000, 0)
        self.assertTrue(np.isinf(icer) and icer > 0)

    def test_calculate_cer(self):
        cer = _calculate_cer(1000, 1.0)
        self.assertEqual(cer, 1000)

        # Zero QALY
        cer = _calculate_cer(1000, 0)
        self.assertTrue(np.isinf(cer) or cer > 1e10)

    def test_validate_model_parameters(self):
        # Should not raise
        _validate_model_parameters(self.model_params)

        # Should raise for missing keys
        with self.assertRaises(ValueError):
            _validate_model_parameters({"states": []})

    def test_calculate_friction_cost(self):
        friction_cost = _calculate_friction_cost(self.model_params, "standard_care")
        # Result can be array or scalar
        self.assertTrue(isinstance(friction_cost, (int, float, np.number, np.ndarray)))

    def test_get_costs_qalys_by_perspective(self):
        costs_sc, costs_nt, qalys_sc, qalys_nt = _get_costs_qalys_by_perspective(
            self.model_params, "health_system", "human_capital"
        )
        self.assertEqual(len(costs_sc), 3)
        self.assertEqual(len(qalys_sc), 3)

    def test_create_parameters_table(self):
        df = create_parameters_table(self.model_params)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_generate_comparative_icer_table(self):
        hs_results = run_cea(self.model_params, perspective="health_system")
        soc_results = run_cea(self.model_params, perspective="societal")
        df = generate_comparative_icer_table(hs_results, soc_results, "Test Intervention")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

if __name__ == "__main__":
    unittest.main()
