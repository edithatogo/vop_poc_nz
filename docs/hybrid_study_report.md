# Codebase Investigator Report: `hybrid_study/`

## Summary of Findings

The `hybrid_study/` directory contains a comprehensive and self-contained health economic evaluation project. Its primary purpose is to synthesize and demonstrate advanced methodologies, chief among them being the direct comparison of 'health system' and 'societal' perspectives in Cost-Effectiveness Analysis (CEA).

**Key Features & Methodologies:**
- **Core Engine:** A flexible `HybridCEAModel` class that implements a three-state Markov model.
- **Dual Perspectives:** The entire framework is designed to run analyses for both 'health system' and 'societal' costs, allowing for direct comparison.
- **Decision Discordance:** A key innovation of this study is a specific analysis (`calculate_decision_discordance`) that quantifies the economic and health (QALY) loss incurred when a decision based on the health system perspective differs from the optimal decision from a societal perspective.
- **Advanced Analyses:** The study includes implementations or stubs for Probabilistic Sensitivity Analysis (PSA), threshold analysis, Budget Impact Analysis (`bia_model.py`), and Expected Value of Perfect Information (`evpi.py`).
- **Automated Reporting:** The project is designed to generate a rich set of outputs, including summary CSV files, publication-quality plots, and detailed, consulting-style reports in Markdown.

**Inputs:**
- The study does not use external data files. The `data/` directory is empty.
- All model parameters for the three case studies (HPV Vaccination, Smoking Cessation, Hepatitis C Therapy) are hard-coded as Python dictionaries within the main execution script, `src/run_hybrid_analysis.py`, and duplicated in the `notebooks/hybrid_study_analysis.ipynb`.

**Workflow:**
- The analysis is driven by `src/run_hybrid_analysis.py`, which initializes the `HybridCEAModel`, runs the analysis for all case studies, and generates all final reports and visualizations in the `results/` directory.
- An accompanying Jupyter notebook provides an interactive, step-by-step version of the same analysis.

This `hybrid_study` represents the most feature-complete version of the analysis in the repository, combining deterministic and probabilistic models with a novel framework for quantifying the value of a societal perspective.

## Exploration Trace

- Used `list_directory` to inspect the contents of `hybrid_study/`.
- Read `hybrid_study/README.md` to get a high-level overview of the project's purpose and features.
- Used `list_directory` to inspect the `hybrid_study/src/` directory to identify key source code files.
- Read `hybrid_study/src/hybrid_cea_model.py` to understand the core logic, data structures, and methodologies of the cost-effectiveness model.
- Used `list_directory` on `hybrid_study/data/` and found it was empty, leading to the hypothesis that inputs are hard-coded.
- Read `hybrid_study/src/run_hybrid_analysis.py` to confirm the hard-coded inputs and understand the main analysis workflow and output generation process.
- Used `list_directory` to inspect the `hybrid_study/notebooks/` directory.
- Read `hybrid_study/notebooks/hybrid_study_analysis.ipynb` to see how it related to the main script, confirming it was an interactive counterpart.
- Read `hybrid_study/requirements.txt` to identify all project dependencies.

## Relevant Locations

- **`hybrid_study/src/hybrid_cea_model.py`**: This is the core of the entire study. It defines the `HybridCEAModel` class, which contains the fundamental logic for the Markov model, the calculation of cost-effectiveness metrics, and the implementation of all key analytical features like dual perspectives, probabilistic analysis, and decision discordance. All other scripts and notebooks rely on this file.
  - Key Symbols: `HybridCEAModel`, `run_deterministic_analysis`, `run_probabilistic_analysis`, `run_threshold_analysis`, `calculate_decision_discordance`, `generate_comprehensive_report`
- **`hybrid_study/src/run_hybrid_analysis.py`**: This is the main execution script for the study. It serves two critical functions: 1) It defines the hard-coded input parameters for all three case studies (HPV, Smoking Cessation, Hep C). 2) It orchestrates the entire analysis pipeline, calling the `HybridCEAModel` and generating all final outputs (CSV data, PNG plots, and a comprehensive Markdown report).
  - Key Symbols: `setup_interventions`, `run_comprehensive_analysis`, `generate_visualizations`, `generate_comprehensive_report`
- **`hybrid_study/notebooks/hybrid_study_analysis.ipynb`**: This notebook serves as an interactive and exploratory version of the `run_hybrid_analysis.py` script. It reuses the same core logic and hard-coded inputs but presents the analysis in a cell-by-cell format, making it useful for demonstration and debugging. It produces slightly different but conceptually identical outputs.
- **`hybrid_study/README.md`**: This file provides an excellent, high-level summary of the study's purpose, the methodologies it synthesizes from previous work, its key innovations (like decision discordance), and its intended outputs. It serves as a vital guide to the project's architecture and goals.
- **`hybrid_study/requirements.txt`**: This file lists the project's dependencies. The presence of standard libraries like `numpy`, `pandas`, and `matplotlib` is expected. The inclusion of `scikit-learn` and `networkx` suggests capabilities for more advanced analyses (like discrete choice experiments) beyond the core CEA model.
