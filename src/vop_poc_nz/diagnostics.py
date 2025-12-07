"""
Diagnostics Module.

This module provides functions to check the stability and convergence of
probabilistic sensitivity analysis (PSA) results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check_psa_convergence(
    psa_results: pd.DataFrame,
    metric_col: str = "incremental_nmb",
    output_path: str = None,
) -> dict:
    """
    Check convergence of a PSA metric by calculating the running mean and standard error.

    Args:
        psa_results: DataFrame containing PSA results (must have 'iteration' or be ordered).
        metric_col: Column name of the metric to check (e.g., 'inc_nmb').
        output_path: Path to save the convergence plot (optional).

    Returns:
        Dict containing convergence statistics.
    """
    if metric_col not in psa_results.columns:
        raise ValueError(f"Column {metric_col} not found in PSA results.")

    values = psa_results[metric_col].values
    n_samples = len(values)
    iterations = np.arange(1, n_samples + 1)

    # Calculate running mean
    running_mean = np.cumsum(values) / iterations

    # Calculate running standard error
    # SE = std / sqrt(n)
    # We can compute running std efficiently or just loop for clarity (n=1000 is small)
    running_std = np.array([np.std(values[:i]) for i in range(1, n_samples + 1)])
    running_se = running_std / np.sqrt(iterations)

    # Confidence Intervals (95%)
    ci_lower = running_mean - 1.96 * running_se
    ci_upper = running_mean + 1.96 * running_se

    # Plotting
    if output_path:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(iterations, running_mean, label="Running Mean", color="blue")
        ax.fill_between(
            iterations, ci_lower, ci_upper, color="blue", alpha=0.2, label="95% CI"
        )
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel(f"Mean {metric_col}")
        ax.set_title(f"PSA Convergence: {metric_col}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Convergence check: Is the change in the last 10% of iterations < 1%?
    last_10_percent = int(n_samples * 0.1)
    mean_end = np.mean(running_mean[-last_10_percent:])
    mean_start_of_end = running_mean[-last_10_percent]
    pct_change = abs((mean_end - mean_start_of_end) / mean_end)

    return {
        "n_samples": n_samples,
        "final_mean": running_mean[-1],
        "final_se": running_se[-1],
        "pct_change_last_10pct": pct_change,
        "converged": pct_change < 0.01,
    }
