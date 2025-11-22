# Comprehensive Documentation for Enhanced Canonical Health Economic Analysis Code

This document provides detailed information about the architecture, features, and usage of the enhanced canonical health economic analysis codebase (`vop_poc_nz/`).

## 1. Architecture

The codebase follows a modular Python project structure, organized to promote reusability, testability, and maintainability.

### Directory Structure

```
vop_poc_nz/
├── src/                    # Source code for core modules
│   ├── cea_model_core.py   # Core Corrected CEA model and helper functions
│   ├── dcea_analysis.py    # Full Discrete Choice Experiment Analysis implementation
│   ├── value_of_information.py  # Probabilistic Sensitivity Analysis and VOI
│   ├── dsa_analysis.py     # Deterministic Sensitivity Analysis implementations
│   ├── discordance_analysis.py # Decision Discordance Analysis
│   ├── threshold_analysis.py # Threshold Analysis implementations
│   ├── bia_model.py        # Budget Impact Analysis model
│   ├── reporting.py        # Automated reporting functions
│   ├── plotting.py         # Bespoke plotting functions for visualizations
│   ├── cluster_analysis.py # Cluster Analysis implementations
│   └── main_analysis.py    # Main script orchestrating all analyses
├── data/                   # Input data files (e.g., empirical DCE data)
├── docs/                   # Project-specific documentation (e.g., protocols)
├── tests/                  # Unit and integration tests
│   ├── test_analysis.py    # Tests for core CEA and VOI
│   └── test_new_features.py # Tests for newly integrated features
├── notebooks/              # Jupyter notebooks for interactive analysis and tutorials
├── output/                 # Output files (reports, plots, tables)
├── requirements.txt        # Python dependencies
├── README.md               # High-level project overview
└── documentation.md        # This detailed documentation
```

### Module Dependencies

-   **`main_analysis.py`**: Orchestrates all other modules.
-   **`cea_model_core.py`**: Provides core CEA functions (`run_cea`, `MarkovModel`) used by `value_of_information.py`, `dsa_analysis.py`, `discordance_analysis.py`, `threshold_analysis.py`, `reporting.py`.
-   **`dcea_analysis.py`**: Independent, but its results are integrated in `main_analysis.py`.
-   **`value_of_information.py`**: Depends on `cea_model_core.py`.
-   **`dsa_analysis.py`**: Depends on `cea_model_core.py`.
-   **`discordance_analysis.py`**: Depends on `cea_model_core.py`.
-   **`threshold_analysis.py`**: Depends on `cea_model_core.py`.
-   **`bia_model.py`**: Independent.
-   **`reporting.py`**: Depends on `cea_model_core.py` and `discordance_analysis.py`.
-   **`plotting.py`**: Depends on `numpy`, `matplotlib`, `seaborn`, and probabilistic results from `value_of_information.py`.
-   **`cluster_analysis.py`**: Depends on `numpy`, `pandas`, `sklearn`.

## 2. Features

The codebase now includes a comprehensive suite of health economic evaluation features:

-   **Corrected CEA Calculations**: Addresses common errors in ICER, NMB, and discounting.
-   **Probabilistic Sensitivity Analysis (PSA)**: Conducts Monte Carlo simulations to quantify uncertainty.
-   **Value of Information (VOI) Analysis**: Calculates EVPI and EVPPI for research prioritization.
-   **Discrete Choice Experiment Analysis (DCEA)**: Quantifies stakeholder preferences using `pylogit` for mixed logit models.
-   **Advanced Deterministic Sensitivity Analysis (DSA)**: Includes comprehensive two-way and three-way DSA with various plotting options.
-   **Decision Discordance Analysis**: Quantifies the economic and health losses due to differing optimal decisions across perspectives.
-   **Threshold Analysis**: Identifies critical parameter values at which policy decisions would switch.
-   **Budget Impact Analysis (BIA)**: Projects the financial consequences of intervention adoption over time.
-   **Cluster Analysis**: Uses K-Means and PCA to identify patterns in cost-effectiveness results and intervention archetypes.
-   **Automated Reporting**: Generates comprehensive Markdown reports summarizing key findings for each intervention.
-   **Bespoke Publication-Grade Plotting**: A dedicated module for generating high-quality visualizations.
-   **Transparency and Documentation**: Adherence to CHEERS 2022 guidelines, detailed parameter tables, and comprehensive code documentation.

## 3. Usage

The main entry point for running the comprehensive analysis is the `main` function in `vop_poc_nz/src/main_analysis.py`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_repository/vop_poc_nz.git
    cd vop_poc_nz
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

To run the complete analysis, execute the `main_analysis.py` script:

```bash
python vop_poc_nz/src/main_analysis.py
```

This script will:

-   Perform corrected CEA for all defined interventions (HPV, smoking cessation, hepatitis C).
-   Generate comparative ICER tables and parameter documentation.
-   Run Value of Information analysis.
-   Perform DCEA (using synthetic or empirical data if `DCE_DATA_PATH` is set).
-   Conduct various Deterministic Sensitivity Analyses (DSA).
-   Perform Cluster Analysis to identify archetypes.
-   Generate all required output plots and save them in `output/figures/`.
-   Generate comprehensive Markdown reports for each intervention in `output/`.

### Output

The analysis generates several important outputs in the `output/` directory:

-   `comparative_icer_table.csv`: Side-by-side comparison of health system vs. societal perspectives.
-   `parameters_assumptions_sources_table.csv`: Full documentation of all model parameters.
-   `voi_analysis_summary.json`: Summary of Value of Information analysis results.
-   `complete_analysis_results.json`: Complete results including all corrections and integrated analyses.
-   `literature_informed_dcea_table.csv`: Preference-weighted NMB view from DCEA.
-   `{intervention_name}_report.md`: Comprehensive Markdown report for each intervention.
-   `figures/`: A directory containing all generated plots in PNG and PDF formats.

### Empirical DCE Data

If you have empirical DCE data, you can set the `DCE_DATA_PATH` environment variable to the path of your CSV file. The `dcea_analysis.py` module will attempt to load and use this data.

```bash
export DCE_DATA_PATH="/path/to/your/dce_data.csv"
python vop_poc_nz/src/main_analysis.py
```

## 4. Testing

Unit and integration tests are located in the `vop_poc_nz/tests/` directory.

-   `test_analysis.py`: Contains tests for core CEA, VOI, and DCEA data processing.
-   `test_new_features.py`: Contains tests for newly integrated features including DSA, Discordance Analysis, Threshold Analysis, BIA, and Reporting.

To run all tests:

```bash
python -m unittest discover vop_poc_nz/tests/
```
