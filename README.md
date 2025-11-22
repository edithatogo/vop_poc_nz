# Canonical Health Economic Analysis Code

This repository contains a comprehensive, corrected implementation of health economic evaluation methods addressing all reviewer feedback from the NZMJ submission. The code implements proper cost-effectiveness analysis with societal perspective considerations, rigorous value of information analysis, and discrete choice experiment analysis.

## Overview

This implementation addresses all the issues identified by reviewers:

1. **Mathematical errors in ICER calculations** - Fixed with proper validation
2. **Transparency of parameters/assumptions/sources** - Full documentation table
3. **Missing comparative ICER table** - Created for side-by-side comparison
4. **EVPPI methodology and justification** - Proper implementation with explanation
5. **Full Discrete Choice Experiment Analysis (DCEA)** - Proper DCE framework
6. **Policy implications expansion** - Detailed analysis of societal vs health system perspectives
7. **CHEERS 2022 compliance** - Full checklist adherence
8. **Analytical capacity costs** - Detailed cost calculations and funding entity identification

## Directory Structure

```
canonical_code/
├── src/                    # Source code
│   ├── cea_model_core.py   # Corrected CEA model with validation
│   ├── dcea_analysis.py    # Full DCEA implementation 
│   ├── value_of_information.py  # Proper VOI analysis
│   ├── parameters.yaml     # Model parameters
│   └── main_analysis.py    # Full integrated analysis
├── data/                   # Data files (if any)
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks for analysis
├── output/                 # Output files (created during execution)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features

### 1. Corrected CEA Calculations
- Fixed mathematical errors in ICER calculations
- Added comprehensive parameter validation
- Implemented proper discounting methodology
- Added detailed documentation for all calculations

### 2. Discrete Choice Experiment Analysis (DCEA)
- Full implementation of DCE methodology
- Experimental design following best practices
- Conditional logit and mixed logit modeling
- Stakeholder preference quantification
- Integration with CEA results

### 3. Rigorous Value of Information Analysis
- Proper EVPI and EVPPI calculations
- Explanation of value even when ICERs are below WTP
- Research prioritization guidance
- Population-level impact assessment

### 4. Advanced Sensitivity Analyses
- Comprehensive two-way Deterministic Sensitivity Analysis (DSA)
- Three-way DSA with 3D visualizations
- Comparative DSA for interventions

### 5. Decision Discordance and Threshold Analysis
- Quantifies decision discordance between perspectives
- Identifies decision-switching points through threshold analysis

### 6. Budget Impact Analysis (BIA)
- Comprehensive projection of financial consequences of adopting new interventions

### 7. Cluster Analysis
- Identifies intervention archetypes and cost-effectiveness patterns using clustering techniques

### 8. Bespoke Plotting Library
- Generates publication-grade plots including CE planes with ellipses, CEAC, CEAF, EVPI, net benefit curves, and various DSA plots.

### 9. Automated Reporting
- Generates comprehensive Markdown reports for each intervention summarizing key findings.

### 10. Transparency and Documentation
- Comprehensive parameters/assumptions/sources table
- Detailed code documentation
- CHEERS 2022 compliance report
- Clear methodology explanations

## Requirements

- Python 3.8+
- See `requirements.txt` for specific package versions

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The model parameters are defined in `src/parameters.yaml`. A template file, `src/parameters.yaml.template`, is provided. To run the analysis with your own parameters, copy the template to `src/parameters.yaml` and modify the values as needed.

The `parameters.yaml` file is ignored by git, so your local changes will not be committed.

## Usage

Run the comprehensive analysis with:

```bash
cd canonical_code/src
python main_analysis.py
```

This will:
- Perform corrected CEA for all interventions defined in `parameters.yaml`
- Generate comparative ICER tables
- Create parameters documentation
- Run value of information analysis
- Perform DCEA (with synthetic data for demonstration)
- Generate all required outputs

## Key Outputs

The analysis generates several important outputs in the `output/` directory:

1. `comparative_icer_table.csv` - Side-by-side comparison of health system vs societal perspectives
2. `parameters_assumptions_sources_table.csv` - Full documentation of all parameters
3. `voi_analysis_summary.json` - Value of information analysis results
4. `complete_analysis_results.json` - Complete results including all corrections

## Addressing Reviewer Feedback

### Critical Issues Fixed:
- **ICER Calculation Errors**: All mathematical errors corrected with validation
- **Parameter Transparency**: Complete parameters table with sources
- **Comparative ICER Table**: Created for direct comparison between perspectives

### Methodological Improvements:
- **DCEA Implementation**: Full discrete choice experiment framework
- **EVPPI Methodology**: Proper probabilistic sensitivity analysis with justification
- **Policy Implications**: Expanded analysis of societal vs health system perspectives
- **CHEERS Compliance**: Full checklist adherence achieved

### Technical Enhancements:
- **Code Validation**: Comprehensive input validation and error checking
- **Documentation**: Detailed docstrings and methodology explanations
- **Reproducibility**: Complete parameter specification and random seed control

## Files Description

- `cea_model_core.py`: Core corrected CEA model with proper mathematical calculations
- `dcea_analysis.py`: Full Discrete Choice Experiment Analysis implementation
- `value_of_information.py`: Proper EVPI/EVPPI calculations with methodology justification
- `main_analysis.py`: Integrated analysis combining all improvements

## Reproducibility

All analyses use fixed random seeds for reproducible results. The implementation follows best practices for health economic evaluation and addresses all reproducibility concerns raised by reviewers.

## Validation

The code includes validation checks at each step to ensure mathematical correctness and prevent the errors identified by reviewers. All parameter inputs are validated before use in calculations.

## Contributing

For questions about the implementation or to report issues, please contact the research team through the submission system.
