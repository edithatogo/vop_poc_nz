"""
vop_poc_nz: Distributional Cost-Effectiveness Analysis Framework.

A comprehensive health economic evaluation framework implementing
Distributional Cost-Effectiveness Analysis (DCEA) with rigorous
value of information methods and global sensitivity analysis.
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Research Team"

# Core CEA functionality
# Budget Impact Analysis
from vop_poc_nz.bia_model import calculate_budget_impact
from vop_poc_nz.cea_model_core import (
    MarkovModel,
    create_parameters_table,
    run_cea,
)

# DCEA equity analysis
from vop_poc_nz.dcea_equity_analysis import (
    calculate_atkinson_index,
    calculate_gini,
    run_dcea,
)

# Sensitivity Analysis
from vop_poc_nz.dsa_analysis import run_dsa, run_two_way_dsa
from vop_poc_nz.sobol_analysis import SobolAnalyzer

# Validation
from vop_poc_nz.validation import validate_psa_results

# Value of Information
from vop_poc_nz.value_of_information import (
    ProbabilisticSensitivityAnalysis,
    calculate_evpi,
    calculate_evppi,
)

__all__ = [
    # CEA
    "MarkovModel",
    # VOI
    "ProbabilisticSensitivityAnalysis",
    "SobolAnalyzer",
    # Version
    "__version__",
    "calculate_atkinson_index",
    # BIA
    "calculate_budget_impact",
    "calculate_evpi",
    "calculate_evppi",
    # DCEA
    "calculate_gini",
    "create_parameters_table",
    "run_cea",
    "run_dcea",
    # Sensitivity
    "run_dsa",
    "run_two_way_dsa",
    # Validation
    "validate_psa_results",
]
