
import pytest
import numpy as np
from src.cea_model_core import run_cea
from src.dcea_equity_analysis import run_dcea, apply_equity_weights

def test_dcea_integration_logic():
    """
    Test that DCEA integration works correctly:
    1. run_cea produces subgroup results.
    2. run_dcea calculates equity metrics.
    3. Equity weighting works as expected.
    """
    # 1. Define parameters with subgroups
    params = {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.1, 0], [0, 0.9, 0.1], [0, 0, 1]],
            "new_treatment": [[0.95, 0.05, 0], [0, 0.95, 0.05], [0, 0, 1]],
        },
        "cycles": 10,
        "initial_population": [1000, 0, 0],
        "costs": {
            "health_system": {
                "standard_care": [100, 1000, 0],
                "new_treatment": [200, 1000, 0],
            },
            "societal": {
                "standard_care": [0, 0, 0],
                "new_treatment": [0, 0, 0],
            }
        },
        "qalys": {
            "standard_care": [1.0, 0.5, 0],
            "new_treatment": [1.0, 0.6, 0],
        },
        "productivity_costs": {
            "human_capital": {
                "standard_care": [0, 0, 0],
                "new_treatment": [0, 0, 0],
            }
        },
        "subgroups": {
            "GroupA": { # Favored group (e.g., higher disease burden or lower SES)
                "initial_population": [1000, 0, 0],
                # Let's say they have worse outcomes in standard care, so they benefit more?
                # Or just same params for simplicity, but we weight them higher.
            },
            "GroupB": { # Less favored group
                "initial_population": [1000, 0, 0],
            }
        }
    }

    # 2. Run CEA
    results = run_cea(params, perspective="societal")
    
    assert "subgroup_results" in results
    assert "GroupA" in results["subgroup_results"]
    assert "GroupB" in results["subgroup_results"]
    
    # Check that we have NMB for subgroups
    nmb_a = results["subgroup_results"]["GroupA"]["incremental_nmb"]
    nmb_b = results["subgroup_results"]["GroupB"]["incremental_nmb"]
    
    # Since params are identical, NMB should be identical
    assert np.isclose(nmb_a, nmb_b)
    
    # 3. Run DCEA with weights
    equity_weights = {
        "GroupA": 2.0, # High priority
        "GroupB": 1.0, # Standard priority
    }
    
    dcea_results = run_dcea(
        results["subgroup_results"], 
        epsilon=0.5, 
        equity_weights=equity_weights
    )
    
    # 4. Verify weighted results
    total_unweighted = dcea_results["total_health_gain"]
    total_weighted = dcea_results["weighted_total_health_gain"]
    
    expected_unweighted = nmb_a + nmb_b
    expected_weighted = (nmb_a * 2.0) + (nmb_b * 1.0)
    
    assert np.isclose(total_unweighted, expected_unweighted)
    assert np.isclose(total_weighted, expected_weighted)
    
    # Weighted should be higher than unweighted (since weights >= 1 and at least one is > 1)
    assert total_weighted > total_unweighted
    
    print(f"Unweighted: {total_unweighted}")
    print(f"Weighted: {total_weighted}")
