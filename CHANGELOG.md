# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2025-11-29

### Added

- **Enhanced Snakefile Template**
  - `configfile` directive for parameters.yaml integration
  - Output versioning with configurable version tag
  - Logging support with `tee` for all rules
  - `discordance_loss.png` output in workflow
  - `clean_all` rule to remove all outputs
  - Test rule with coverage reporting

### Changed

- **Publication Quality Figures**
  - Increased default DPI from 300 to 1200 for all visualizations
  - Updated DSA plots (tornado, heatmaps, 3D surfaces)
  - Updated comparative visualizations (cash flow, ICER ladder, NMB, equity)

## [0.1.2] - 2025-11-29

### Added

- **Project Scaffolding**
  - `vop-poc-nz init` - Initialize project with Snakefile and parameters template
  - `--force` flag to overwrite existing files
  - Bundled Snakefile template for Snakemake workflow integration

## [0.1.1] - 2025-11-29

### Added

- **Command-Line Interface (CLI)**
  - `vop-poc-nz run` - Run full analysis pipeline (CEA, DCEA, VOI, DSA, reporting)
  - `vop-poc-nz report` - Generate reports from previously saved results
  - `--output-dir` / `-o` flag for custom output directory
  - `--skip-reporting` flag to run analysis only
  - `--version` flag to display version

### Fixed

- Fixed missing `validate_psa_results` import in `value_of_information.py`
- Fixed missing `load_parameters` import in `profile_scalability.py`
- Improved pyright/mypy configuration for src-layout compatibility

## [0.1.0] - 2025-11-29

### Added

- **Core CEA Framework**
  - `MarkovModel` class for health state transitions with proper validation
  - `run_cea()` function for cost-effectiveness analysis
  - Support for both health system and societal perspectives
  - Proper discounting of costs and QALYs

- **Distributional Cost-Effectiveness Analysis (DCEA)**
  - `calculate_gini()` - Gini coefficient for inequality measurement
  - `calculate_atkinson_index()` - Atkinson index with configurable inequality aversion
  - `run_dcea()` - Full distributional analysis with equity weighting
  - Lorenz curve and equity impact plane visualizations

- **Value of Information Analysis**
  - `ProbabilisticSensitivityAnalysis` class for Monte Carlo simulation
  - `calculate_evpi()` - Expected Value of Perfect Information
  - `calculate_evppi()` - Expected Value of Partial Perfect Information
  - CEAC and CEAF curve generation

- **Sensitivity Analysis**
  - `run_dsa()` - One-way deterministic sensitivity analysis
  - `run_two_way_dsa()` - Two-way sensitivity analysis
  - `SobolAnalyzer` - Global sensitivity analysis using Sobol indices
  - Tornado diagram visualizations

- **Budget Impact Analysis**
  - `calculate_budget_impact()` - Multi-year budget projections
  - Support for implementation costs and offsets

- **Validation & Quality**
  - Input validation with pandera schemas
  - Property-based testing with Hypothesis
  - 95%+ test coverage
  - Type hints throughout

- **Visualization**
  - Publication-quality figures with matplotlib
  - Optional plotnine support for ggplot2-style graphics
  - Cost-effectiveness planes, acceptability curves, tornado diagrams

### Fixed

- ICER calculation edge cases (zero denominators)
- Transition matrix validation
- Discounting formula corrections

### Security

- Added pip-audit and bandit security scanning

[Unreleased]: https://github.com/edithatogo/vop_poc_nz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edithatogo/vop_poc_nz/releases/tag/v0.1.0
