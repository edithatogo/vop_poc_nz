import unittest

import numpy as np
import pandas as pd

from vop_poc_nz.value_of_information import (
    ProbabilisticSensitivityAnalysis,
    calculate_evpi,
    calculate_evppi,
    calculate_population_evpi,
    calculate_value_of_perspective,
    explain_value_of_information_benefits,
    generate_voi_report,
)


class TestValueOfInformation(unittest.TestCase):
    def setUp(self):
        # Simple model function for testing
        def test_model(params, intervention_type="standard_care"):
            if intervention_type == "standard_care":
                return params.get("cost_sc", 1000), params.get("qaly_sc", 10)
            else:
                return params.get("cost_nt", 1500), params.get("qaly_nt", 12)

        self.model_func = test_model
        self.parameters = {
            "cost_sc": {"distribution": "gamma", "params": {"shape": 10, "scale": 100}},
            "cost_nt": {"distribution": "gamma", "params": {"shape": 15, "scale": 100}},
            "qaly_sc": {"distribution": "beta", "params": {"alpha": 50, "beta": 5}},
            "qaly_nt": {"distribution": "beta", "params": {"alpha": 55, "beta": 5}}
        }

        # Create sample PSA results for testing
        np.random.seed(42)
        n = 100
        self.psa_results = pd.DataFrame({
            "cost_sc": np.random.gamma(10, 100, n),
            "cost_nt": np.random.gamma(15, 100, n),
            "qaly_sc": np.random.beta(50, 5, n),
            "qaly_nt": np.random.beta(55, 5, n),
            "inc_cost": np.random.normal(500, 100, n),
            "inc_qaly": np.random.normal(2, 0.5, n),  # Note: inc_qaly not inc_qalys
            "icer": np.random.normal(250, 50, n),
            "nmb": np.random.normal(99500, 1000, n)
        })
        self.psa_results["cost_effective"] = self.psa_results["nmb"] > 0

    def test_psa_init(self):
        psa = ProbabilisticSensitivityAnalysis(self.model_func, self.parameters)
        self.assertEqual(psa.wtp_threshold, 50000)
        self.assertEqual(len(psa.parameters), 4)

    def test_psa_sample_parameters(self):
        psa = ProbabilisticSensitivityAnalysis(self.model_func, self.parameters)
        samples = psa.sample_parameters(n_samples=10)
        self.assertEqual(len(samples), 10)
        self.assertIn("cost_sc", samples[0])
        self.assertIn("qaly_nt", samples[0])

    def test_psa_run(self):
        psa = ProbabilisticSensitivityAnalysis(self.model_func, self.parameters)
        results = psa.run_psa(n_samples=50)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 50)
        # Check for new column naming convention (inc_cost or inc_cost_hs/inc_cost_soc)
        has_inc_cost = "inc_cost" in results.columns or "inc_cost_hs" in results.columns
        self.assertTrue(has_inc_cost, "Expected inc_cost or inc_cost_hs in columns")
        # Check for NMB columns (inc_nmb or inc_nmb_hs/inc_nmb_soc)
        has_nmb = "inc_nmb" in results.columns or "inc_nmb_hs" in results.columns
        self.assertTrue(has_nmb, "Expected inc_nmb or inc_nmb_hs in columns")
        self.assertIn("cost_effective", results.columns)

    def test_psa_calculate_ceac(self):
        psa = ProbabilisticSensitivityAnalysis(self.model_func, self.parameters)
        ceac = psa.calculate_ceac(self.psa_results, wtp_values=[20000, 50000, 100000])
        self.assertIsInstance(ceac, pd.DataFrame)
        self.assertEqual(len(ceac), 3)
        self.assertIn("wtp_threshold", ceac.columns)
        self.assertIn("probability_cost_effective", ceac.columns)

    def test_calculate_evpi(self):
        evpi = calculate_evpi(self.psa_results, wtp_threshold=50000)
        self.assertIsInstance(evpi, (int, float, np.number))
        self.assertGreaterEqual(evpi, 0)

    def test_calculate_evppi(self):
        evppi = calculate_evppi(
            self.psa_results,
            parameter_group=["cost_nt"],
            all_params=["cost_sc", "cost_nt", "qaly_sc", "qaly_nt"],
            wtp_thresholds=[50000],
            n_bootstrap=10
        )
        self.assertIsInstance(evppi, list)
        self.assertGreaterEqual(len(evppi), 0)

    def test_calculate_value_of_perspective(self):
        # Create properly structured PSA results for dual-perspective analysis
        # calculate_value_of_perspective expects: qaly_nt, qaly_sc, cost_nt, cost_sc
        np.random.seed(42)
        n = 100
        psa_hs = pd.DataFrame({
            "cost_sc": np.random.gamma(10, 100, n),
            "cost_nt": np.random.gamma(15, 100, n),
            "qaly_sc": np.random.beta(50, 5, n) * 10,  # Scale to reasonable QALY
            "qaly_nt": np.random.beta(55, 5, n) * 10,
        })
        psa_soc = pd.DataFrame({
            "cost_sc": np.random.gamma(10, 110, n),  # Slightly different costs
            "cost_nt": np.random.gamma(15, 110, n),
            "qaly_sc": np.random.beta(50, 5, n) * 10,
            "qaly_nt": np.random.beta(55, 5, n) * 10,
        })

        vop = calculate_value_of_perspective(
            psa_hs, psa_soc, wtp_threshold=50000, chosen_perspective="health_system"
        )
        self.assertIsInstance(vop, dict)
        self.assertIn("expected_value_of_perspective", vop)
        self.assertIn("perspective_premium", vop)

    def test_calculate_population_evpi(self):
        pop_evpi = calculate_population_evpi(evpi_per_person=100, target_population_size=100000)
        self.assertEqual(pop_evpi, 10000000)

    def test_explain_voi_benefits(self):
        explanation = explain_value_of_information_benefits(base_icer=30000, wtp_threshold=50000)
        self.assertIsInstance(explanation, dict)
        self.assertIn("base_case_info", explanation)

    def test_generate_voi_report(self):
        # generate_voi_report expects specific columns: inc_cost, inc_qaly,
        # cost_sc, cost_nt, qaly_sc, qaly_nt
        report = generate_voi_report(
            self.psa_results,
            wtp_thresholds=[50000],
            target_population=100000,
            parameter_names=["cost_sc", "cost_nt", "qaly_sc", "qaly_nt"]
        )
        self.assertIsInstance(report, dict)
        self.assertIn("summary_statistics", report)
        self.assertIn("value_of_information", report)

if __name__ == "__main__":
    unittest.main()
