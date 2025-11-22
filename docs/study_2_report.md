# Codebase Investigator Report: `study_2_focused_cea/`

## Summary of Findings

The investigation reveals that the `vop_poc_nz` directory is a direct, refactored, and feature-enhanced evolution of the work in `study_2_focused_cea`. The `study_2_focused_cea` project is an exploratory research workbench contained in a single monolithic script (`generate_combined_plots.py`). It features a wide array of advanced analyses, including multi-way deterministic sensitivity analysis (DSA), comparative DSA between interventions, and cluster analysis to identify 'intervention archetypes'.

The `vop_poc_nz` codebase is a 'publication-ready' version, refactored into a modular Python package to address formal journal reviewer feedback.

**Key Features and Methodologies in `study_2_focused_cea`:**
- **Core Model:** 3-state Markov Model for Cost-Effectiveness Analysis (CEA).
- **Perspectives:** Health System and Societal.
- **Probabilistic Analysis (PSA):** Monte Carlo simulations, Cost-Effectiveness Planes (with ellipses), CEAC, and CEAF.
- **Value of Information (VOI):** EVPI, Population EVPI, and EVPPI.
- **Advanced Exploratory Analyses:**
    - Two-way and three-way DSA with heatmap and 3D visualizations.
    - **Comparative DSA:** Directly comparing interventions against each other.
    - **Cluster Analysis:** Using K-Means and PCA to find patterns in simulation results.
    - **Discordance Analysis:** Quantifying the 'Value of Perspective'.

**Comparison with `vop_poc_nz` (Canonical Codebase):**

The `vop_poc_nz` codebase streamlines the analysis by removing the most complex, exploratory features and adding new ones focused on transparency and addressing specific reviewer requests.

**Features Missing in `vop_poc_nz` (that were in `study_2_focused_cea`):**
- The advanced, multi-way, and comparative **Deterministic Sensitivity Analyses (DSA)** have been removed.
- The **Cluster Analysis** for identifying intervention archetypes has been removed.
- The extensive library of bespoke, publication-ready plotting functions has been removed in favor of generating data tables and summary files.

**New Features Added in `vop_poc_nz` (not in `study_2_focused_cea`):**
- **Discrete Choice Experiment Analysis (DCEA):** This is the most significant new feature, implemented in `dcea_analysis.py`. It quantifies stakeholder preferences and integrates them with CEA results.
- **Formal Reporting and Transparency:** New functions were added to generate tables for model parameters/sources (`create_parameters_table`) and to report compliance with the CHEERS 2022 standard (`generate_cheers_report`).
- **Calculation of Analytical Capacity Costs:** A new function (`calculate_analytical_capacity_costs`) was added to estimate the real-world cost of implementing a societal perspective, directly addressing a reviewer comment.
- **Modular Structure:** The code was refactored from a single script into a maintainable package with separate modules for the core model, VOI, and DCEA.

## Exploration Trace

- Started by listing the contents of the `study_2_focused_cea` directory to get an overview of the project structure.
- Read the `README.md` file in `study_2_focused_cea` to understand its high-level purpose and key features, which included various forms of cost-effectiveness analysis.
- Listed the contents of the `study_2_focused_cea/src` directory and identified `generate_combined_plots.py` as the likely main script.
- Read the file `generate_combined_plots.py` and discovered it was a monolithic script containing the entire analysis pipeline, including Markov models, PSA, advanced DSA, VOI, and cluster analysis. This provided a comprehensive feature map for the project.
- Listed the contents of the `vop_poc_nz` directory, noting its more formal Python package structure.
- Read the `README.md` file in `vop_poc_nz`, which revealed it was a 'corrected' and 'refactored' version of the analysis, built to address journal reviewer feedback and introducing a new 'Discrete Choice Experiment Analysis' (DCEA) feature.
- Listed the contents of the `vop_poc_nz/src` directory to confirm its modular structure (`cea_model_core.py`, `dcea_analysis.py`, `value_of_information.py`, `main_analysis.py`).
- Read the `main_analysis.py` script to understand how the refactored modules were integrated and to identify which features from the original script were kept, which were removed, and what new functionalities were added.
- Synthesized the findings from both codebases to perform a detailed comparison, leading to the final summary of findings.

## Relevant Locations

- **`study_2_focused_cea/src/generate_combined_plots.py`**: This monolithic script is the complete implementation of the `study_2_focused_cea` project. It contains a wide array of advanced, exploratory analyses that serve as the primary basis for comparison. Its features represent the project's full analytical capability before it was refactored.
  - Key Symbols: `MarkovModel`, `ProbabilisticCEA`, `perform_comprehensive_two_way_dsa`, `perform_three_way_dsa`, `ClusterAnalysis`, `perform_comparative_two_way_dsa`
- **`vop_poc_nz/src/main_analysis.py`**: This is the main orchestrator script for the canonical codebase. It shows which features from the original study were kept and, crucially, how new features like DCEA and formal reporting (CHEERS, parameter tables) are integrated. It demonstrates the shift in focus from exploration to publication-readiness.
  - Key Symbols: `run_corrected_analysis`, `perform_dcea_analysis`, `perform_voi_analysis`, `generate_cheers_report`, `calculate_analytical_capacity_costs`
- **`vop_poc_nz/src/dcea_analysis.py`**: This file contains the implementation of the Discrete Choice Experiment Analysis (DCEA), which is the most significant new feature added to the canonical codebase (`vop_poc_nz`) and is entirely absent from `study_2_focused_cea`. It represents a major methodological addition.
  - Key Symbols: `DCEAnalyzer`, `DCEDataProcessor`, `integrate_dce_with_cea`, `generate_literature_informed_dcea_view`
- **`vop_poc_nz/src/value_of_information.py`**: This module represents the refactored and cleaned-up version of the probabilistic sensitivity and value of information analyses that were originally embedded in the monolithic `generate_combined_plots.py` script. It shows how the core methods were preserved but modularized.
  - Key Symbols: `ProbabilisticSensitivityAnalysis`, `calculate_evpi`, `calculate_evppi`
