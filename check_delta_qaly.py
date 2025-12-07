import pickle
import numpy as np
import pandas as pd
import os

def check_delta():
    results_path = "output/results.pkl"
    if not os.path.exists(results_path):
        print("Results file not found.")
        return

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    prob_results = results.get("probabilistic_results", {})
    
    for name, df in prob_results.items():
        if "inc_qaly_soc" in df.columns and "inc_qaly_hs" in df.columns:
            delta_qaly = df["inc_qaly_soc"] - df["inc_qaly_hs"]
            if np.allclose(delta_qaly, 0):
                print(f"[{name}] Delta QALY is effectively ZERO.")
            else:
                max_diff = np.max(np.abs(delta_qaly))
                print(f"[{name}] Delta QALY is NOT zero. Max diff: {max_diff}")
        else:
            print(f"[{name}] Missing columns for delta check.")

if __name__ == "__main__":
    check_delta()
