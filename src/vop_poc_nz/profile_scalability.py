"""
Scalability Profiling Script

Tests how PSA performance scales with sample size.
Useful for determining optimal sample sizes for production runs.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import psutil

from .pipeline.analysis import load_parameters
from .value_of_information import ProbabilisticSensitivityAnalysis
from .visualizations import apply_default_style, save_figure


def run_scalability_test(
    sample_sizes: list[int] = None,
    intervention_name: str = "HPV Vaccination",
    output_dir: str = "output/profiling/",
):
    """
    Run PSA with increasing sample sizes and measure time/memory.
    """
    if sample_sizes is None:
        sample_sizes = [100, 500, 1000, 2000, 5000]
    print(f"Running scalability test for {intervention_name}...")
    print(f"Sample sizes: {sample_sizes}")

    # Load parameters
    params = load_parameters()
    model_params = params["hpv_vaccination"]  # Use HPV as benchmark

    results = {
        "n_samples": [],
        "time_seconds": [],
        "memory_mb": [],
        "samples_per_second": [],
    }

    # Define simple model wrapper for PSA
    def psa_model_wrapper(sampled_params, intervention_type):
        # Simplified model run for speed - just to test PSA overhead
        # In a real test, this would call the full Markov model
        # But here we want to test the PSA machinery itself + model overhead

        # Simulate some computational work
        # time.sleep(0.0001)

        if intervention_type == "standard_care":
            return (1000, 10)
        else:
            return (1200, 11)

    # Use the actual PSA class
    psa = ProbabilisticSensitivityAnalysis(
        model_func=psa_model_wrapper,
        parameters=model_params.get("psa_params", {}),  # Need to ensure these exist
        wtp_threshold=50000,
    )

    # If psa_params missing, create dummy ones for testing
    if not psa.parameters:
        psa.parameters = {
            "cost_mult": {
                "distribution": "normal",
                "params": {"mean": 1.0, "std": 0.1},
            },
            "qaly_mult": {
                "distribution": "normal",
                "params": {"mean": 1.0, "std": 0.1},
            },
        }

    os.makedirs(output_dir, exist_ok=True)

    process = psutil.Process(os.getpid())

    for n in sample_sizes:
        print(f"  Testing n={n}...", end="\r")

        # Force garbage collection
        import gc

        gc.collect()

        mem_before = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Run PSA
        psa.run_psa(n_samples=n)

        duration = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_peak = mem_after  # Approximation

        results["n_samples"].append(n)
        results["time_seconds"].append(duration)
        results["memory_mb"].append(mem_peak - mem_before)
        results["samples_per_second"].append(n / duration if duration > 0 else 0)

    print("\n  Test complete.")

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "scalability_results.csv"), index=False)

    # Plot results
    plot_scalability(df, output_dir)

    return df


def plot_scalability(df: pd.DataFrame, output_dir: str):
    """Plot scalability metrics."""
    apply_default_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Time vs Samples
    ax1.plot(df["n_samples"], df["time_seconds"], "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("PSA Execution Time Scaling")
    ax1.grid(True, alpha=0.3)

    # Memory vs Samples
    ax2.plot(df["n_samples"], df["memory_mb"], "s-", color="forestgreen", linewidth=2)
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Memory Usage (MB)")
    ax2.set_title("PSA Memory Usage Scaling")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "psa_scalability_profile")
    plt.close(fig)


if __name__ == "__main__":
    run_scalability_test()
