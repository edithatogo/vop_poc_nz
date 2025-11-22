# Summary of the Canonical Codebase (`vop_poc_nz/`)

This document summarizes the features, inputs, and architecture of the canonical codebase located in the `vop_poc_nz/` directory.

## Architecture

The codebase follows a standard Python project structure:

- **`src/`**: Contains the core logic of the analysis.
  - `cea_model_core.py`: Corrected Cost-Effectiveness Analysis (CEA) model.
  - `dcea_analysis.py`: Full Discrete Choice Experiment Analysis (DCEA) implementation.
  - `value_of_information.py`: Value of Information (VOI) analysis.
  - `main_analysis.py`: Integrates all analyses.
- **`data/`**: For input data files.
- **`docs/`**: For documentation.
- **`tests/`**: For unit tests.
- **`notebooks/`**: For Jupyter notebooks for analysis.
- **`output/`**: For output files generated during execution.
- **Configuration files**: `pyproject.toml`, `setup.cfg`, `pytest.ini`, `ruff.toml` for project management, testing, and linting.
- **Scripts**: `run_basic_text_analysis.py` and `run_textstat_analysis.py` for text analysis.

## Features

The canonical codebase includes the following key features:

1.  **Corrected Cost-Effectiveness Analysis (CEA):**
    -   Addresses mathematical errors in ICER calculations.
    -   Includes comprehensive parameter validation.
    -   Implements proper discounting methodology.

2.  **Discrete Choice Experiment Analysis (DCEA):**
    -   Full implementation of the DCEA methodology.
    -   Includes experimental design, conditional logit and mixed logit modeling.
    -   Quantifies stakeholder preferences.
    -   Integrates with CEA results.

3.  **Value of Information (VOI) Analysis:**
    -   Proper EVPI and EVPPI calculations.
    -   Provides research prioritization guidance.
    -   Assesses population-level impact.

4.  **Transparency and Documentation:**
    -   Provides a comprehensive table of parameters, assumptions, and sources.
    -   Adheres to CHEERS 2022 compliance.

5.  **Text Analysis:**
    -   Includes scripts for basic text analysis and `textstat` analysis.

## Inputs

The primary inputs to the analysis are:

-   **Model Parameters:** The analysis is parameterized through various inputs, which are documented in the `parameters_assumptions_sources_table.csv` output file.
-   **Synthetic Data:** The DCEA is demonstrated using synthetic data.
-   **Text Data:** The text analysis scripts likely take text files as input.

## Outputs

The analysis generates the following outputs in the `output/` directory:

-   `comparative_icer_table.csv`: Compares health system vs. societal perspectives.
-   `parameters_assumptions_sources_table.csv`: Documents all parameters.
-   `voi_analysis_summary.json`: Summarizes VOI analysis results.
-   `complete_analysis_results.json`: Contains complete results of all analyses.
