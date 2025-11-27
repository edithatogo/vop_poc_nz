import pandas as pd
import numpy as np

def calculate_evpi(psa_results: pd.DataFrame, wtp_threshold: float = 50000) -> float:
    # Calculate NMB for each strategy at the given WTP threshold
    nmb_sc = (psa_results["qaly_sc"] * wtp_threshold) - psa_results["cost_sc"]
    nmb_nt = (psa_results["qaly_nt"] * wtp_threshold) - psa_results["cost_nt"]

    # Stack NMBs for each strategy to find the optimal for each simulation
    nmb_matrix = np.column_stack([nmb_sc, nmb_nt])

    # Find maximum NMB across all strategies for each simulation (perfect info scenario)
    max_nmb_per_sim = np.max(nmb_matrix, axis=1)

    # Find the expected NMB with current information (current optimal strategy)
    current_optimal_nmb = np.mean(
        np.max(
            [
                np.mean(nmb_sc),  # Expected NMB of standard care
                np.mean(nmb_nt),  # Expected NMB of new treatment
            ]
        )
    )
    
    print(f"NMB SC Mean: {np.mean(nmb_sc)}")
    print(f"NMB NT Mean: {np.mean(nmb_nt)}")
    print(f"Current Optimal NMB: {current_optimal_nmb}")
    print(f"Max NMB per sim (mean): {np.mean(max_nmb_per_sim)}")

    # EVPI = Expected value with perfect information - Expected value with current info
    evpi = np.mean(max_nmb_per_sim) - current_optimal_nmb
    return evpi

# Test case from failure
data = {"cost_sc": [1000], "qaly_sc": [5], "cost_nt": [900], "qaly_nt": [6]}
df = pd.DataFrame(data)
wtp = 50000

result = calculate_evpi(df, wtp)
print(f"EVPI: {result}")
