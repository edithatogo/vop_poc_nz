"""
vop_poc_nz: Distributional Cost-Effectiveness Analysis Framework.

A comprehensive health economic evaluation framework implementing
Distributional Cost-Effectiveness Analysis (DCEA) with rigorous
value of information methods and global sensitivity analysis.
"""

from __future__ import annotations

__version__ = "0.1.3"
__author__ = "Research Team"

# Core CEA functionality
from vop_poc_nz.cea_model_core import (
    MarkovModel,
    run_cea,
    create_parameters_table,
)

# DCEA equity analysis
from vop_poc_nz.dcea_equity_analysis import (
    calculate_gini,
    calculate_atkinson_index,
    run_dcea,
)

# Value of Information
from vop_poc_nz.value_of_information import (
    ProbabilisticSensitivityAnalysis,
    calculate_evpi,
    calculate_evppi,
)

# Sensitivity Analysis
from vop_poc_nz.dsa_analysis import run_dsa, run_two_way_dsa
from vop_poc_nz.sobol_analysis import SobolAnalyzer

# Budget Impact Analysis
from vop_poc_nz.bia_model import calculate_budget_impact

# Validation
from vop_poc_nz.validation import validate_psa_results

__all__ = [
    # Version
    "__version__",
    # CEA
    "MarkovModel",
    "run_cea",
    "create_parameters_table",
    # DCEA
    "calculate_gini",
    "calculate_atkinson_index",
    "run_dcea",
    # VOI
    "ProbabilisticSensitivityAnalysis",
    "calculate_evpi",
    "calculate_evppi",
    # Sensitivity
    "run_dsa",
    "run_two_way_dsa",
    "SobolAnalyzer",
    # BIA
    "calculate_budget_impact",
    # Validation
    "validate_psa_results",
]
