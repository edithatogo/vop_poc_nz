import json
import numpy as np
import pandas as pd
import os

def inspect_voi():
    results_path = "output/complete_analysis_results.json"
    if not os.path.exists(results_path):
        print("Results file not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    prob_results = results.get("probabilistic_results", {})
    voi_results = results.get("voi_analysis", {})
    
    print("\n--- Debugging Structure ---")
    if prob_results:
        first_key = list(prob_results.keys())[0]
        print(f"Prob Results Keys: {list(prob_results.keys())}")
        print(f"First Item Keys ({first_key}): {list(prob_results[first_key].keys())}")
        
    if voi_results:
        print(f"VOI Results Keys: {list(voi_results.keys())}")
        # Check if it's nested by intervention
        first_voi_key = list(voi_results.keys())[0]
        if isinstance(voi_results[first_voi_key], dict):
             print(f"First VOI Item Keys ({first_voi_key}): {list(voi_results[first_voi_key].keys())}")

if __name__ == "__main__":
    inspect_voi()
