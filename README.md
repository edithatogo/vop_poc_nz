# Value of Perspective: Quantifying Decision Discordance in Health Economic Evaluation

[![CI](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edithatogo/vop_poc_nz/branch/main/graph/badge.svg)](https://codecov.io/gh/edithatogo/vop_poc_nz)
[![PyPI](https://img.shields.io/pypi/v/vop-poc-nz)](https://pypi.org/project/vop-poc-nz/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17759272.svg)](https://doi.org/10.5281/zenodo.17759272)
[![OSF](https://img.shields.io/badge/OSF-unq76-blue)](https://osf.io/unq76)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A proof-of-concept framework for quantifying the **Value of Perspective (VoP)** in health economic evaluation. This project uses published economic evaluations from the Aotearoa New Zealand context to measure the decision impact of choosing between health system and societal perspectives.

## The Value of Perspective Concept

When decision-makers evaluate health interventions, the choice of analytical perspective can lead to different conclusions:

- **Health System Perspective**: Includes only direct healthcare costs (hospitalisations, medications, clinical visits)
- **Societal Perspective**: Additionally includes productivity losses, caregiver burden, and broader social costs

The **Value of Perspective (VoP)** quantifies the potential loss from decision discordance—when the two perspectives yield conflicting cost-effectiveness conclusions. This serves as:

1. **A trigger threshold**: Identifying when perspective choice materially affects recommendations
2. **A quantitative measure**: Expressing opportunity cost in monetary terms (NZ$/QALY lost)

## Key Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Core Analysis** | Cost-Effectiveness Analysis | Validated Markov cohort models with proper discounting and half-cycle correction |
| | Perspective Comparison | Side-by-side health system vs societal ICER calculations |
| | Decision Discordance | Automated detection of conflicting recommendations |
| | Value of Perspective | Quantified loss from adopting narrower perspective |
| **Equity Analysis** | Distributional CEA | Gini coefficient and Atkinson index (multiple ε values) |
| | Subgroup Analysis | Recursive CEA across population segments |
| | Equity Weighting | Customisable weights for disadvantaged groups |
| **Uncertainty** | Probabilistic SA | Monte Carlo simulation with distribution sampling |
| | EVPI/EVPPI | Expected value of perfect (partial) information |
| | Sobol Analysis | Variance-based global sensitivity indices |
| | Threshold Analysis | Decision-critical parameter ranges |
| **Visualisation** | CE Planes | Incremental cost-effectiveness scatter plots |
| | CEAC/CEAF | Acceptability curves and frontiers |
| | Tornado Diagrams | One-way, two-way, and three-way DSA |
| | Lorenz Curves | Health inequality visualisation |
| **Reporting** | CHEERS 2022 | Consolidated Health Economic Evaluation Reporting Standards |
| | Parameter Tables | Transparent documentation of assumptions and sources |
| | Policy Briefs | Executive-level decision summaries |
| **Infrastructure** | Budget Impact Analysis | Multi-year projections (1-10 years) with discounting |
| | Snakemake Pipeline | Reproducible workflow orchestration |
| | Comprehensive Testing | 184 tests with 95%+ coverage |

## Case Studies from Aotearoa New Zealand

This framework was developed using published economic evaluations relevant to the NZ healthcare context:

1. **HPV Vaccination** - Societal benefits include productivity gains from cancer prevention
2. **Smoking Cessation** - Broader costs captured through absenteeism and family impact
3. **Hepatitis C Treatment** - Reduced transmission creates societal value beyond direct healthcare

## Quick Start

### Installation

```bash
pip install vop-poc-nz
```

### Basic Usage

```python
from vop_poc_nz import run_cea

# Run analysis from both perspectives
health_system = run_cea(params, perspective="health_system")
societal = run_cea(params, perspective="societal")

# Calculate Value of Perspective
from vop_poc_nz.discordance_analysis import calculate_decision_discordance

discordance = calculate_decision_discordance(
    "HPV Vaccination",
    params,
    wtp_threshold=50000
)

if discordance["discordant"]:
    print(f"Decision discordance detected!")
    print(f"Value of Perspective: ${discordance['loss_from_discordance']:,.0f}")
    print(f"Equivalent to {discordance['loss_qaly']:.2f} QALYs lost")
```

### Snakemake Pipeline

For reproducible full analysis:

```bash
snakemake -c1
```

## Directory Structure

```
vop_poc_nz/
├── src/vop_poc_nz/               # Source package
│   ├── cea_model_core.py         # CEA Markov model
│   ├── discordance_analysis.py   # Value of Perspective calculations
│   ├── dcea_equity_analysis.py   # Distributional CEA
│   ├── value_of_information.py   # EVPI/EVPPI analysis
│   ├── sobol_analysis.py         # Global sensitivity
│   ├── dsa_analysis.py           # Deterministic sensitivity analysis
│   ├── bia_model.py              # Budget impact analysis
│   ├── visualizations.py         # Publication-quality plots
│   └── parameters.yaml           # Model parameters
├── tests/                        # Unit tests (pytest)
├── docs/                         # Documentation
├── output/                       # Generated outputs
└── Snakefile                     # Workflow definition
```

## Key Outputs

### Tables (CSV)
- `comparative_icer_table.csv` - Health System vs Societal perspective comparison
- `value_of_perspective_table.csv` - Discordance metrics by intervention
- `parameters_assumptions_sources_table.csv` - Full transparency documentation
- `dcea_equity_results.csv` - Subgroup analysis with equity metrics

### Figures (PNG/PDF/SVG)
- Perspective comparison planes
- Decision discordance heatmaps
- CEAC/CEAF curves by perspective
- Tornado diagrams (1-way, 2-way, 3-way DSA)
- Lorenz curves and equity impact planes

## Methodological References

- **CEA**: Drummond et al. (2015) - Methods for the Economic Evaluation of Health Care Programmes
- **Perspectives**: Sanders et al. (2016) - Recommendations for Conduct of Cost-Effectiveness Analysis (Second Panel)
- **DCEA**: Cookson et al. (2017) - Distributional Cost-Effectiveness Analysis
- **VOI**: Claxton et al. (2001) - The Value of Information
- **CHEERS**: Husereau et al. (2022) - CHEERS 2022 Reporting Guidelines

For detailed mathematical specifications, see [FORMULAE.md](docs/FORMULAE.md).

## Testing

```bash
pytest                              # Run all tests
pytest --cov=src --cov-report=html  # With coverage
```

Test coverage: **184 tests passing** with **95%+ coverage**

## CI/CD

GitHub Actions workflows:
- **CI** (`.github/workflows/ci.yml`): Testing across Python 3.10-3.13, linting, type checking
- **Release** (`.github/workflows/release.yml`): PyPI publishing via Trusted Publishers (OIDC)

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{mordaunt_2024_17759272,
  author       = {Mordaunt, Dylan A},
  title        = {Value of Perspective: Quantifying Decision Discordance in Health Economic Evaluation},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17759272},
  url          = {https://doi.org/10.5281/zenodo.17759272}
}
```

### Research Data & Materials

- **Zenodo Archive**: [10.5281/zenodo.17759272](https://doi.org/10.5281/zenodo.17759272) - Citable software archive with DOI
- **OSF Project**: [osf.io/unq76](https://osf.io/unq76) - Supplementary materials and data

## Acknowledgments

- Developed using published economic evaluations from the Aotearoa New Zealand context
- Inspired by ISPOR and PHARMAC guidelines for health economic evaluation
