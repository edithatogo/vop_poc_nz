
import pickle
import numpy as np
import pandas as pd

def inspect_data():
    print("Loading results...")
    with open("output/results.pkl", "rb") as f:
        results = pickle.load(f)
        
    print("\n--- Inspecting Markov Traces ---")
    if 'intervention_results' in results:
        for name, res in results['intervention_results'].items():
            print(f"\nIntervention: {name}")
            # Traces are likely in 'health_system' or 'societal' sub-dictionaries
            if 'health_system' in res:
                hs_res = res['health_system']
                if 'trace_standard_care' in hs_res:
                    trace = hs_res['trace_standard_care']
                    print(f"  Standard Care Trace Shape: {trace.shape}")
                    if isinstance(trace, pd.DataFrame):
                        print(f"  Standard Care Trace Head:\n{trace.head()}")
                        row_sums = trace.sum(axis=1)
                    else:
                        print(f"  Standard Care Trace Head:\n{trace[:5]}")
                        row_sums = trace.sum(axis=1)
                    
                    print(f"  Row Sums (should be 1.0): {row_sums[:5]}")
                    if not np.allclose(row_sums, 1.0):
                        print("  WARNING: Standard Care rows do not sum to 1.0!")
                else:
                    print("  Standard Care Trace MISSING in health_system")
                    
                if 'trace_new_treatment' in hs_res:
                    trace = hs_res['trace_new_treatment']
                    print(f"  New Treatment Trace Shape: {trace.shape}")
                    if isinstance(trace, pd.DataFrame):
                        row_sums = trace.sum(axis=1)
                    else:
                        row_sums = trace.sum(axis=1)
                        
                    print(f"  Row Sums (should be 1.0): {row_sums[:5]}")
                    if not np.allclose(row_sums, 1.0):
                        print("  WARNING: New Treatment rows do not sum to 1.0!")
                else:
                    print("  New Treatment Trace MISSING in health_system")
            else:
                print("  'health_system' key MISSING in intervention result")

    print("\n--- Inspecting EVPI Data ---")
    if 'probabilistic_results' in results:
        for name, df in results['probabilistic_results'].items():
            print(f"\nIntervention: {name}")
            print(f"  PSA Results Columns: {df.columns.tolist()}")
            print(f"  PSA Results Head:\n{df.head()}")
            
            # Check for zero variance which would lead to zero EVPI
            if 'net_benefit_hs' in df.columns:
                print(f"  Net Benefit HS Std Dev: {df['net_benefit_hs'].std()}")
            if 'net_benefit_soc' in df.columns:
                print(f"  Net Benefit Soc Std Dev: {df['net_benefit_soc'].std()}")

if __name__ == "__main__":
    inspect_data()
