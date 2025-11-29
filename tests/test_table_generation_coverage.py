import os
import shutil
import tempfile
import unittest

import pandas as pd

from vop_poc_nz.table_generation import (
    dataframe_to_markdown,
    generate_all_tables,
    generate_bia_tables,
    generate_cea_tables,
    generate_core_tables,
    generate_dcea_tables,
    generate_extended_bia_tables,
    generate_extended_dcea_tables,
    generate_extended_voi_tables,
    generate_qualitative_tables,
    generate_subgroup_tables,
    generate_voi_tables,
)


class TestTableGenerationCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Dummy Params
        self.params = {
            "Intervention_A": {
                "discount_rate": 0.03,
                "states": ["Healthy", "Sick", "Dead"],
                "initial_population": [900, 100, 0],
                "costs": {
                    "health_system": {"new_treatment": [100, 200, 0]},
                    "societal": {"new_treatment": [50, 100, 0]}
                },
                "qalys": {"new_treatment": [1.0, 0.7, 0.0]},
                "dsa_parameter_ranges": {
                    "cost_drug": {"base": 100, "range": "80-120", "source": "Pharma"}
                },
                "treatment_effect": {"value": 0.8, "source": "Trial"},
                "subgroups": {
                    "Maori": {"population_size": 1000, "weight": 1.5, "param_x": 10},
                    "Non-Maori": {"population_size": 5000, "weight": 1.0}
                },
                "bia_population": {"total_population": 5000000, "eligible_proportion": 0.01}
            },
            "Intervention_B": {
                # Missing treatment_effect to test else branch
                "states": ["A"],
                "initial_population": [0], # Test zero pop
                "costs": {}, "qalys": {}
            }
        }

        # Dummy Results
        self.results = {
            "intervention_results": {
                "Intervention_A": {
                    "health_system": {
                        "cost_new_treatment": 1000, "qalys_new_treatment": 10,
                        "incremental_cost": 500, "incremental_qalys": 0.5,
                        "icer": 1000, "incremental_nmb": 20000
                    },
                    "societal": {
                        "human_capital": {
                            "cost_new_treatment": 1200, "qalys_new_treatment": 10,
                            "incremental_cost": 600, "incremental_qalys": 0.5,
                            "icer": 1200, "incremental_nmb": 19000
                        }
                    }
                }
            },
            "dsa_analysis": {
                "1_way": {
                    "Intervention_A": {
                        "health_system": {
                            "param_1": {"low_input": 0.8, "high_input": 1.2, "range": 5000}
                        }
                    }
                }
            },
            "probabilistic_results": {
                "Intervention_A": {
                    "health_system": {
                        "incremental_costs": [400, 600],
                        "incremental_qalys": [0.4, 0.6]
                    }
                }
            },
            "dcea_equity_analysis": {
                "Intervention_A": {
                    "societal": {
                        "distribution_of_net_health_benefits": {"Maori": 100, "Non-Maori": 200},
                        "outcomes_by_group": {
                            "Maori": {"incremental_qalys": 0.1, "incremental_costs": 100, "net_health_benefit": 50}
                        },
                        "equity_metrics": {
                            "atkinson_index": 0.1,
                            "health_achievement_index": 0.9,
                            "equity_weighted_nmb": 25000
                        }
                    },
                    "health_system": {
                        "distribution_of_net_health_benefits": {"Maori": 80, "Non-Maori": 180}
                    }
                }
            },
            "bia_results": {
                "Intervention_A": pd.DataFrame({
                    "year": [1, 2, 3, 4, 5],
                    "gross_cost": [100, 100, 100, 100, 100],
                    "net_cost": [80, 80, 80, 80, 80]
                })
            },
            "voi_analysis": {
                "Intervention_A": {
                    "value_of_information": {
                        "population_evpi": 500000,
                        "evppi_by_parameter_group": {
                            "Group A": [10, 20],
                            "Group B": 15.5 # Scalar case
                        },
                        "wtp_thresholds": [20000, 50000]
                    }
                }
            }
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataframe_to_markdown(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        md = dataframe_to_markdown(df)
        self.assertIsInstance(md, str)
        # Check for column headers loosely
        self.assertTrue("A" in md and "B" in md)

    def test_generate_core_tables(self):
        tables = generate_core_tables(self.params, self.test_dir)
        self.assertIn("Table_1_Study_Setting", tables)

        # Test skip logic
        params_skip = {"new_high_cost_cancer_drug": {}}
        tables_skip = generate_core_tables(params_skip, self.test_dir)
        # Should produce empty tables or tables with no rows, but not crash
        self.assertIn("Table_2_Interventions", tables_skip)

    def test_generate_qualitative_tables(self):
        tables = generate_qualitative_tables(self.params, self.test_dir)
        self.assertIn("Table_4_Treatment_Effects", tables)

    def test_generate_cea_tables(self):
        tables = generate_cea_tables(self.results, self.test_dir)
        self.assertIn("Table_10_Base_Case_CEA", tables)

    def test_generate_subgroup_tables(self):
        tables = generate_subgroup_tables(self.results, self.test_dir)
        self.assertIn("Table_15_Subgroup_Results", tables)

    def test_generate_dcea_tables(self):
        tables = generate_dcea_tables(self.results, self.test_dir)
        self.assertIn("Table_19_DCEA_Outcomes_by_Group", tables)

        # Test empty results
        tables_empty = generate_dcea_tables({}, self.test_dir)
        self.assertEqual(tables_empty, {})

    def test_generate_extended_dcea_tables(self):
        tables = generate_extended_dcea_tables(self.results, self.params, self.test_dir)
        self.assertIn("Table_16_Equity_Population", tables)

    def test_generate_bia_tables(self):
        tables = generate_bia_tables(self.results, self.test_dir)
        self.assertIn("Table_24_Annual_Budget_Impact", tables)

        # Test list of dicts input
        results_list = {
            "bia_results": {
                "Int_B": [{"year": 1, "gross_cost": 10, "net_cost": 5}]
            }
        }
        tables_list = generate_bia_tables(results_list, self.test_dir)
        self.assertIn("Table_24_Annual_Budget_Impact", tables_list)

        # Test empty
        self.assertEqual(generate_bia_tables({}, self.test_dir), {})

    def test_generate_extended_bia_tables(self):
        tables = generate_extended_bia_tables(self.results, self.params, self.test_dir)
        self.assertIn("Table_22_BIA_Target_Population", tables)

    def test_generate_voi_tables(self):
        tables = generate_voi_tables(self.results, self.test_dir)
        self.assertIn("Table_27_EVPI_Summary", tables)

        # Test legacy format
        results_legacy = {
            "voi_analysis": {
                "Int_Legacy": {
                    "evpi": {"thresholds": [50000], "evpi": [1000]},
                    "evppi": {"Param": 100}
                }
            }
        }
        tables_legacy = generate_voi_tables(results_legacy, self.test_dir)
        self.assertIn("Table_27_EVPI_Summary", tables_legacy)
        self.assertIn("Table_28_EVPPI_Summary", tables_legacy)

        # Test empty
        self.assertEqual(generate_voi_tables({}, self.test_dir), {})

    def test_generate_extended_voi_tables(self):
        tables = generate_extended_voi_tables(self.results, self.test_dir)
        self.assertIn("Table_29_EVSI", tables)

    def test_generate_all_tables(self):
        tables = generate_all_tables(self.results, self.params, self.test_dir)
        self.assertTrue(len(tables) > 0)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "generated_tables.md")))

if __name__ == "__main__":
    unittest.main()
