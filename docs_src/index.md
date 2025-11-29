# Health Economic Analysis: Distributional Cost-Effectiveness Framework

[![CI](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edithatogo/vop_poc_nz/branch/main/graph/badge.svg)](https://codecov.io/gh/edithatogo/vop_poc_nz)
[![PyPI](https://img.shields.io/pypi/v/vop-poc-nz)](https://pypi.org/project/vop-poc-nz/)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/edithatogo/vop_poc_nz/blob/main/LICENSE)

Comprehensive health economic evaluation framework implementing **Distributional Cost-Effectiveness Analysis (DCEA)** with rigorous value of information methods and global sensitivity analysis.

## Features

- **Cost-Effectiveness Analysis (CEA)** - Validated Markov cohort models with proper discounting
- **Distributional CEA (DCEA)** - Equity analysis using Gini and Atkinson indices
- **Value of Information (VOI)** - EVPI/EVPPI for research prioritization
- **Global Sensitivity Analysis** - Sobol variance-based methods
- **Budget Impact Analysis (BIA)** - Multi-year projections with discounting
- **Comprehensive Reporting** - CHEERS 2022 compliant outputs

## Quick Start

```bash
pip install vop-poc-nz
```

```python
from vop_poc_nz import run_cea, run_dcea, calculate_evpi

# Load your parameters
import yaml
with open("parameters.yaml") as f:
    params = yaml.safe_load(f)

# Run CEA
results = run_cea(params["intervention"], perspective="societal")

# Check if cost-effective
if results["icer"] < 50000:
    print(f"Cost-effective: ICER = ${results['icer']:,.0f}/QALY")
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](guide/cea.md)
- [API Reference](api/cea_model_core.md)
- [Methodology](methodology/formulae.md)

## License

Apache License 2.0 - see [LICENSE](https://github.com/edithatogo/vop_poc_nz/blob/main/LICENSE) for details.
