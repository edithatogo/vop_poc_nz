# Health Economic Analysis: Distributional Cost-Effectiveness Framework

[![CI](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edithatogo/vop_poc_nz/branch/main/graph/badge.svg)](https://codecov.io/gh/edithatogo/vop_poc_nz)
[![PyPI](https://img.shields.io/pypi/v/vop-poc-nz)](https://pypi.org/project/vop-poc-nz/)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Comprehensive health economic evaluation framework implementing **Distributional Cost-Effectiveness Analysis (DCEA)** with rigorous value of information methods and global sensitivity analysis.

## Overview

This implementation addresses the complete spectrum of health economic evaluation with a focus on equity considerations:

1. **Cost-Effectiveness Analysis (CEA)** - Validated Markov cohort models with proper discounting
2. **Distributional CEA (DCEA)** - Equity analysis using Gini and Atkinson indices
3. **Value of Information (VOI)** - EVPI/EVPPI for research prioritization
4. **Global Sensitivity Analysis** - Sobol variance-based methods
5. **Budget Impact Analysis (BIA)** - Multi-year projections with discounting
6. **Comprehensive Reporting** - CHEERS 2022 compliant outputs

## Architecture

![Architecture](docs/diagrams/architecture.mmd)

The codebase is modular with clear separation between:
- **Core Analysis**: CEA, DCEA, VOI, DSA modules
- **Pipeline**: Orchestration and workflow management
- **Visualization**: Publication-quality plotting
- **Reporting**: Automated report generation

See [architecture diagrams](docs/diagrams/) for detailed module dependencies and data flow.

## Key Features

### ✅ Distributional Cost-Effectiveness Analysis
- **Equity Metrics**: Gini coefficient, Atkinson index (multiple ε values)
- **Subgroup Analysis**: Automatic recursive CEA for population segments
- **Equity Weighting**: Customizable weights for disadvantaged groups
- **Visualization**: Lorenz curves, equity impact planes, inequality sensitivity

### ✅ Value of Information
- **EVPI**: Expected Value of Perfect Information using two-level Monte Carlo
- **EVPPI**: Parameter-specific information value
- **Population EVPI**: Time-discounted research value
- **CEAC/CEAF**: Cost-effectiveness acceptability curves and frontiers

### ✅ Sobol Sensitivity Analysis  
- **Global SA**: Variance-based indices using Saltelli sampling
- **First-order indices**: Main parameter effects
- **Total-order indices**: Including interactions
- **No external dependencies**: Custom implementation (no SALib required)

### ✅ Enhanced Visualizations
- **Acceptability Frontier**: Optimal intervention at each WTP threshold
- **Population EVPI Timeline**: Research value decay over time
- **Threshold Waterfall**: Decision-critical parameter ranges
- **Multi-Interventionradar**: Trade-off visualization across dimensions

### ✅ Budget Impact Analysis
- **Multi-year projections** (1-10 years)
- **Discounted costs** (customizable discount rate)
- **Implementation costs** (one-time year 1 expenses)
- **Net budget impact** (gross costs - offsets)

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd vop_poc_nz

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run full analysis pipeline
snakemake -c1

# Run with specific version
snakemake -c1 --config version=v2.0

# Run tests via Tox (Recommended)
tox                  # Run all environments (tests, lint, type, coverage)
tox -e py313         # Run tests on Python 3.13
tox -e lint          # Run linting
tox -e type          # Run type checking

# Run memory profiling
memray run -o output/memray.bin -m src.pipeline.reporting
memray flamegraph output/memray.bin -o output/memray-flamegraph.html

# Run linting manually
ruff check .
ruff format .
```

### Python API

```python
from src.cea_model_core import run_cea
from src.dcea_equity_analysis import run_dcea
from src.sobol_analysis import SobolAnalyzer

# Load parameters
import yaml
from importlib import resources

with resources.files("vop_poc_nz").joinpath("parameters.yaml").open("r") as f:
    params = yaml.safe_load(f)

# Run CEA with subgroups
results = run_cea(
    params["hpv_vaccination"],
    perspective="societal",
    productivity_cost_method="human_capital"
)

# Perform DCEA if subgroups exist
if results["subgroup_results"]:
    equity = run_dcea(
        results["subgroup_results"],
        epsilon=0.5,
        equity_weights={"Low_SES": 1.5, "High_SES": 1.0}
    )
    print(f"Atkinson Index: {equity['atkinson_index']:.3f}")
    print(f"Gini Coefficient: {equity['gini_coefficient']:.3f}")

# Run Sobol analysis
def model_wrapper(params):
    cea_results = run_cea(params, perspective="health_system")
    return cea_results["incremental_nmb"]

sobol = SobolAnalyzer(model_wrapper, param_distributions, n_samples=500)
indices = sobol.calculate_sobol_indices()
print(indices['indices'])
```

## Directory Structure

```
vop_poc_nz/
├── src/vop_poc_nz/               # Source package
│   ├── pipeline/                 # Analysis orchestration
│   │   ├── analysis.py           # Core pipeline logic
│   │   └── reporting.py          # Report generation
│   ├── cea_model_core.py         # CEA Markov model
│   ├── dcea_equity_analysis.py   # Distributional CEA
│   ├── value_of_information.py   # VOI analysis
│   ├── sobol_analysis.py         # Sobol sensitivity
│   ├── dsa_analysis.py           # Deterministic SA
│   ├── bia_model.py              # Budget impact
│   ├── visualizations.py         # Core plotting
│   ├── visualizations_extended.py # Additional plots
│   ├── parameters.yaml           # Model parameters
│   └── main.py                   # CLI entry point
├── tests/                        # Unit tests (pytest)
├── docs/                         # Documentation
│   ├── diagrams/                 # Mermaid diagrams (.mmd)
│   └── TUTORIAL.md               # Step-by-step guide
├── output/                       # Generated outputs
├── Snakefile                     # Workflow definition
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Key Outputs

All outputs are saved to `output/latest/` (symlinked to `output/{version}/`):

### Tables (CSV)
- `comparative_icer_table.csv` - Health System vs Societal perspectives
- `parameters_assumptions_sources_table.csv` - Full transparency documentation
- `dcea_equity_results.csv` - Subgroup analysis with equity metrics
- `sobol_indices.csv` - Global sensitivity analysis results

### Figures (PNG/PDF/SVG)
- Cost-effectiveness planes
- Tornado diagrams (1-way, 2-way, 3-way DSA)
- CEAC/CEAF curves
- Lorenz curves and equity impact planes
- Sobol sensitivity bar charts
- Acceptability frontiers
- Population EVPI timelines
- Multi-intervention radar plots

### Reports (Markdown)
- `combined_report.md` - Comprehensive analysis summary
- `policy_brief.md` - Executive-level summary
- `cheers_compliance.md` - CHEERS 2022 checklist

## Methodological References

For detailed mathematical specifications, see [FORMULAE.md](docs/FORMULAE.md).

- **CEA**: Drummond et al. (2015) - Methods for the Economic Evaluation of Health Care Programmes
- **DCEA**: Cookson et al. (2017) - Distributional Cost-Effectiveness Analysis
- **Atkinson Index**: Atkinson (1970) - On the Measurement of Inequality
- **VOI**: Claxton et al. (2001) - The Value of Information
- **Sobol Analysis**: Saltelli et al. (2010) - Variance-Based Sensitivity Analysis
- **CHEERS**: Husereau et al. (2022) - CHEERS 2022 Reporting Guidelines

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_dcea_equity_smoke.py

# Run with verbose output
pytest -v
```

Test coverage: **163 tests passing** with **95%+ coverage**

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:
1. **Testing**: Multi-version Python matrix (3.10-3.13)
2. **Linting**: `ruff check` and `ruff format --check`
3. **Type checking**: `mypy` with strict settings
4. **Coverage**: Uploaded to Codecov with 95% target
5. **Security**: `pip-audit` and `bandit` scanning
6. **Pre-commit**: Validates all hooks pass

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dcea_framework_2024,
  title={Distributional Cost-Effectiveness Analysis Framework},
  author={[Your Name]},
  year={2024},
  url={[repository-url]}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Acknowledgments

- Inspired by ISPOR guidelines for health economic evaluation
- Built with support from [funding sources]
- Thanks to reviewers whose feedback improved this implementation
