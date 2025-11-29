"""
vop_poc_nz: Value of Perspective - Quantifying Decision Discordance.

A proof-of-concept framework for quantifying the Value of Perspective (VoP)
in health economic evaluation. Uses published economic evaluations from
Aotearoa New Zealand to measure decision impact when choosing between
health system and societal perspectives.

The Value of Perspective (VoP) quantifies the potential loss from decision
discordance—when different analytical perspectives yield conflicting
cost-effectiveness conclusions. VoP serves as both:

1. A trigger threshold for when perspective choice materially affects decisions
2. A quantitative measure expressing opportunity cost (NZ$/QALY lost)

Key modules:
    - cea_model_core: Cost-effectiveness analysis with Markov models
    - discordance_analysis: Value of Perspective calculations
    - dcea_equity_analysis: Distributional CEA with equity metrics
    - value_of_information: EVPI/EVPPI analysis
    - sobol_analysis: Global sensitivity analysis
"""

from __future__ import annotations

__version__ = "0.1.6"
__author__ = "Dylan A Mordaunt"

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

# Discordance / Value of Perspective
from vop_poc_nz.discordance_analysis import calculate_decision_discordance

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
    # Value of Perspective / Discordance
    "calculate_decision_discordance",
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
