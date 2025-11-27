"""
Sobol Sensitivity Analysis Module

Custom implementation of Sobol' variance-based global sensitivity analysis
without external dependencies (no SALib required).

Implements:
- Sobol sequence generation (quasi-random sampling)
- First-order sensitivity indices (S_i)
- Total-order sensitivity indices (S_Ti)
- Saltelli sampling scheme

References:
- Saltelli et al. (2010) - Variance-based sensitivity analysis
- Sobol' (2001) - Global sensitivity indices
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


class SobolAnalyzer:
    """
    Sobol variance-based global sensitivity analysis.

    Uses Saltelli's sampling scheme to efficiently compute first-order
    and total-order sensitivity indices showing parameter importance
    and interactions.
    """

    def __init__(
        self,
        model_func: Callable,
        param_distributions: Dict[str, Dict],
        n_samples: int = 1000,
    ):
        """
        Initialize Sobol analyzer.

        Args:
            model_func: Model function that takes dict of parameters and returns scalar output
            param_distributions: Dict of parameter distributions (same format as PSA)
            n_samples: Number of base samples (total evaluations = n_samples * (2*n_params + 2))
        """
        self.model_func = model_func
        self.param_distributions = param_distributions
        self.n_samples = n_samples
        self.param_names = list(param_distributions.keys())
        self.n_params = len(self.param_names)

    def generate_sobol_sequence(self, dimensions: int, n: int) -> np.ndarray:
        """
        Generate Sobol sequence for quasi-random sampling.

        Simplified implementation using van der Corput sequence.
        For production use, consider scipy.stats.qmc.Sobol.

        Args:
            dimensions: Number of dimensions
            n: Number of samples

        Returns:
            Array of shape (n, dimensions) with values in [0, 1]
        """
        # Use Latin Hypercube Sampling as a simpler alternative
        # For true Sobol, would use scipy.stats.qmc.Sobol
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dimensions, scramble=True)
        samples = sampler.random(n=n)
        return samples

    def sample_parameters(self, samples_01: np.ndarray) -> pd.DataFrame:
        """
        Transform [0,1] samples to parameter distributions.

        Args:
            samples_01: Array of shape (n, n_params) with values in [0, 1]

        Returns:
            DataFrame with sampled parameter values
        """
        from scipy.stats import beta, gamma, norm, uniform

        n_samples = samples_01.shape[0]
        sampled_params = {}

        for i, param_name in enumerate(self.param_names):
            dist_info = self.param_distributions[param_name]
            dist_type = dist_info["distribution"]
            params = dist_info["params"]

            # Get uniform samples for this parameter
            u = samples_01[:, i]

            # Transform to target distribution using inverse CDF
            if dist_type == "beta":
                values = beta.ppf(u, params["alpha"], params["beta"])
            elif dist_type == "gamma":
                values = gamma.ppf(u, params["shape"], scale=params["scale"])
            elif dist_type == "normal":
                values = norm.ppf(u, params["mean"], params["std"])
            elif dist_type == "uniform":
                values = uniform.ppf(u, params["low"], params["high"] - params["low"])
            else:
                raise ValueError(f"Unknown distribution: {dist_type}")

            sampled_params[param_name] = values

        return pd.DataFrame(sampled_params)

    def saltelli_sampling(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
        """
        Generate Saltelli sample matrices for Sobol analysis.

        Returns:
            Tuple of (matrix_A, matrix_B, matrices_AB)
            - matrix_A: Base sample matrix (n_samples, n_params)
            - matrix_B: Resample matrix (n_samples, n_params)
            - matrices_AB: List of mixed matrices, one per parameter
        """
        # Generate two independent Sobol sequences
        samples_A = self.generate_sobol_sequence(self.n_params, self.n_samples)
        samples_B = self.generate_sobol_sequence(self.n_params, self.n_samples)

        # Transform to parameter distributions
        matrix_A = self.sample_parameters(samples_A)
        matrix_B = self.sample_parameters(samples_B)

        # Generate AB matrices (replace i-th column of A with i-th column of B)
        matrices_AB = []
        for i in range(self.n_params):
            matrix_AB_i = matrix_A.copy()
            matrix_AB_i.iloc[:, i] = matrix_B.iloc[:, i]
            matrices_AB.append(matrix_AB_i)

        return matrix_A, matrix_B, matrices_AB

    def evaluate_model(self, param_matrix: pd.DataFrame) -> np.ndarray:
        """
        Evaluate model for all parameter sets in matrix.

        Args:
            param_matrix: DataFrame with parameter values

        Returns:
            Array of model outputs
        """
        outputs = []
        for idx in range(len(param_matrix)):
            params = param_matrix.iloc[idx].to_dict()
            try:
                output = self.model_func(params)
                # Handle tuple outputs (e.g., (cost, qaly))
                if isinstance(output, tuple):
                    output = output[0] - output[1] * 50000  # Convert to NMB
                outputs.append(output)
            except Exception:
                # Handle model failures gracefully
                outputs.append(np.nan)

        return np.array(outputs)

    def calculate_sobol_indices(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate Sobol sensitivity indices using Saltelli sampling.

        Returns:
            Dict with:
            - 'first_order': DataFrame with first-order indices (S_i)
            - 'total_order': DataFrame with total-order indices (S_Ti)
            - 'confidence_intervals': DataFrame with 95% CI
        """
        print(f"\\nGenerating Saltelli samples ({self.n_samples} base samples)...")
        matrix_A, matrix_B, matrices_AB = self.saltelli_sampling()

        print(f"Evaluating model (N={(2 + self.n_params) * self.n_samples} runs)...")
        # Evaluate model for all matrices
        Y_A = self.evaluate_model(matrix_A)
        Y_B = self.evaluate_model(matrix_B)
        Y_AB = [self.evaluate_model(mat) for mat in matrices_AB]

        # Calculate variance
        var_Y = np.var(np.concatenate([Y_A, Y_B]))

        # Calculate first-order indices
        first_order = []
        total_order = []

        for i in range(self.n_params):
            # First-order index: S_i = V[E(Y|X_i)] / V(Y)
            # Estimated as: Cov(Y_A, Y_AB_i) / V(Y)
            cov_i = np.mean(Y_B * (Y_AB[i] - Y_A))
            S_i = cov_i / var_Y if var_Y > 0 else 0
            first_order.append(S_i)

            # Total-order index: S_Ti = E[V(Y|X_~i)] / V(Y)
            # Estimated as: 1 - Cov(Y_A, Y_AB_i) / V(Y)
            # Or equivalently: 0.5 * E[(Y_A - Y_AB_i)^2] / V(Y)
            var_conditional = 0.5 * np.mean((Y_A - Y_AB[i]) ** 2)
            S_Ti = var_conditional / var_Y if var_Y > 0 else 0
            total_order.append(S_Ti)

        # Create results DataFrames
        indices_df = pd.DataFrame(
            {
                "parameter": self.param_names,
                "first_order": first_order,
                "total_order": total_order,
                "interaction": [
                    total - first for total, first in zip(total_order, first_order)
                ],
            }
        )

        # Bootstrap confidence intervals (simplified)
        n_bootstrap = 100
        first_order_boot = []
        total_order_boot = []

        for _ in range(n_bootstrap):
            # Resample indices
            boot_idx = np.random.choice(len(Y_A), len(Y_A), replace=True)
            Y_A_boot = Y_A[boot_idx]
            Y_B_boot = Y_B[boot_idx]
            Y_AB_boot = [Y_AB[i][boot_idx] for i in range(self.n_params)]

            var_Y_boot = np.var(np.concatenate([Y_A_boot, Y_B_boot]))

            S_i_boot = []
            S_Ti_boot = []
            for i in range(self.n_params):
                cov_i = np.mean(Y_B_boot * (Y_AB_boot[i] - Y_A_boot))
                S_i_boot.append(cov_i / var_Y_boot if var_Y_boot > 0 else 0)

                var_cond = 0.5 * np.mean((Y_A_boot - Y_AB_boot[i]) ** 2)
                S_Ti_boot.append(var_cond / var_Y_boot if var_Y_boot > 0 else 0)

            first_order_boot.append(S_i_boot)
            total_order_boot.append(S_Ti_boot)

        first_order_boot = np.array(first_order_boot)
        total_order_boot = np.array(total_order_boot)

        ci_df = pd.DataFrame(
            {
                "parameter": self.param_names,
                "first_order_ci_low": np.percentile(first_order_boot, 2.5, axis=0),
                "first_order_ci_high": np.percentile(first_order_boot, 97.5, axis=0),
                "total_order_ci_low": np.percentile(total_order_boot, 2.5, axis=0),
                "total_order_ci_high": np.percentile(total_order_boot, 97.5, axis=0),
            }
        )

        return {
            "indices": indices_df,
            "confidence_intervals": ci_df,
            "n_evaluations": (2 + self.n_params) * self.n_samples,
        }


def plot_sobol_indices(
    sobol_results: Dict,
    output_dir: str = "output/figures/",
    title: str = "Sobol Sensitivity Indices",
):
    """
    Plot Sobol sensitivity indices.

    Args:
        sobol_results: Results from SobolAnalyzer.calculate_sobol_indices()
        output_dir: Output directory
        title: Plot title
    """
    import matplotlib.pyplot as plt

    indices_df = sobol_results["indices"]
    ci_df = sobol_results["confidence_intervals"]

    # Sort by total-order index
    indices_df = indices_df.sort_values("total_order", ascending=True)
    params = indices_df["parameter"].values

    fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.4)), dpi=300)

    y_pos = np.arange(len(params))

    # Plot total-order indices (includes interactions)
    ax.barh(
        y_pos,
        indices_df["total_order"],
        height=0.4,
        label="Total-order ($S_{Ti}$)",
        color="steelblue",
        alpha=0.7,
    )

    # Plot first-order indices (main effects only)
    ax.barh(
        y_pos,
        indices_df["first_order"],
        height=0.4,
        label="First-order ($S_i$)",
        color="darkblue",
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel("Sensitivity Index")
    ax.set_title(f"{title}\\n(Total Evaluations: {sobol_results['n_evaluations']:,})")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max(1, max(indices_df["total_order"]) * 1.1))

    # Add interaction annotation
    for i, (param, interaction) in enumerate(zip(params, indices_df["interaction"])):
        if interaction > 0.05:  # Only annotate significant interactions
            ax.annotate(
                f"Interaction: {interaction:.2f}",
                xy=(indices_df.iloc[i]["total_order"], i),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    plt.tight_layout()

    # Save figure
    import os

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ["png", "pdf"]:
        filepath = os.path.join(output_dir, f"sobol_indices.{fmt}")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"\\nSaved Sobol indices plot to {output_dir}")
