# Codebase Investigator Report: `study_1_nzmj_proof_of_concept/`

## Summary of Findings

The investigation of `study_1_nzmj_proof_of_concept/` is complete. It represents a simplified, proof-of-concept analysis.

**Key Features & Methodology of `study_1_nzmj_proof_of_concept/`:**
- **Methodology:** It uses a deterministic, 3-state Markov model for cost-effectiveness analysis. The core logic is in `cea_model_simple.py`.
- **Inputs:** All model parameters for three case studies (HPV, Smoking, Hepatitis C) are hardcoded directly into `run_study_1_analysis.py`. There are no external input data files.
- **Perspectives:** The analysis is explicitly designed to compare a 'health_system' perspective with a 'societal' perspective. The societal perspective is implemented by adding a separate 'societal' cost component (representing productivity losses) to the health system costs.
- **Outputs:** The primary outputs are Incremental Cost-Effectiveness Ratios (ICER) and Net Monetary Benefit (NMB) for each case study and perspective, saved to JSON and CSV files.

To complete the original objective, the next step would be to perform an equivalent investigation of the `vop_poc_nz/` directory and then compare its features, inputs, and methodologies against these findings. Based on the `README`, the canonical code is expected to be more advanced, likely featuring probabilistic sensitivity analysis, more complex models, and potentially external data sources.

## Exploration Trace

- Used `list_directory` to explore the contents of `study_1_nzmj_proof_of_concept/`.
- Read `study_1_nzmj_proof_of_concept/README.md` to get a high-level overview.
- Used `list_directory` to see the source files in `study_1_nzmj_proof_of_concept/src/`.
- Read `study_1_nzmj_proof_of_concept/src/run_study_1_analysis.py` to understand the main workflow and input parameters.
- Read `study_1_nzmj_proof_of_concept/src/cea_model_simple.py` to understand the core deterministic Markov model.
- Used `list_directory` to view the contents of the `notebooks` and `data` directories, confirming the workflow and lack of external data inputs.
- Attempted to read documentation from the `docs` directory to finalize understanding.

## Relevant Locations

- **`study_1_nzmj_proof_of_concept/src/run_study_1_analysis.py`**: This is the main script for the study. It defines the input parameters for the three case studies (HPV, Smoking Cessation, Hepatitis C) and orchestrates the analysis by calling the CEA model. It explicitly runs the analysis from both 'health_system' and 'societal' perspectives. All inputs are hardcoded here.
  - Key Symbols: `run_hpv_vaccination_analysis`, `run_smoking_cessation_analysis`, `run_hepatitis_c_analysis`, `run_all_case_studies`
- **`study_1_nzmj_proof_of_concept/src/cea_model_simple.py`**: This file contains the core scientific methodology. The `SimpleMarkovModel` class implements a deterministic, discrete-time Markov model for cohort simulation. The `run_simple_cea` function encapsulates the logic for running the model for standard care vs. a new treatment, calculating incremental costs, QALYs, ICER, and NMB. It handles the logic for switching between health system and societal cost perspectives.
  - Key Symbols: `SimpleMarkovModel`, `run_simple_cea`
- **`study_1_nzmj_proof_of_concept/README.md`**: Provides a clear, high-level summary of the study's purpose, methodology (deterministic, 3-state Markov model), and key differences from the more advanced 'Study 2', which is likely the canonical codebase.
