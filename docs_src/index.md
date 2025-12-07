# Value of Perspective: Quantifying Decision Discordance in Health Economic Evaluation

[![CI](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edithatogo/vop_poc_nz/branch/main/graph/badge.svg)](https://codecov.io/gh/edithatogo/vop_poc_nz)
[![PyPI](https://img.shields.io/pypi/v/vop-poc-nz)](https://pypi.org/project/vop-poc-nz/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17759272.svg)](https://doi.org/10.5281/zenodo.17759272)
[![OSF](https://img.shields.io/badge/OSF-unq76-blue)](https://osf.io/unq76)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/edithatogo/vop_poc_nz/blob/main/LICENSE)

A proof-of-concept framework for quantifying the **Value of Perspective (VoP)** in health economic evaluation. This project uses published economic evaluations from the Aotearoa New Zealand context to measure the decision impact of choosing between health system and societal perspectives.

## The Value of Perspective Concept

The **Value of Perspective (VoP)** quantifies the potential loss from decision discordance—when different analytical perspectives yield conflicting cost-effectiveness conclusions:

- **Health System Perspective**: Direct healthcare costs only
- **Societal Perspective**: Additionally includes productivity losses, caregiver burden, and broader costs

VoP serves as both:

1. **A trigger threshold**: Identifying when perspective choice materially affects recommendations
2. **A quantitative measure**: Expressing opportunity cost in monetary terms (NZ$/QALY lost)

## Key Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Core Analysis** | Cost-Effectiveness Analysis | Markov models with discounting |
| | Perspective Comparison | Health system vs societal ICERs |
| | Value of Perspective | Quantified loss from discordance |
| **Equity** | Distributional CEA | Gini, Atkinson indices |
| **Uncertainty** | EVPI/EVPPI | Value of information analysis |
| | Sobol Analysis | Global sensitivity indices |
| **Reporting** | CHEERS 2022 | Standards compliance |

## Quick Start

```bash
pip install vop-poc-nz
```

```python
from vop_poc_nz import run_cea
from vop_poc_nz.discordance_analysis import calculate_decision_discordance

# Compare perspectives
discordance = calculate_decision_discordance("Intervention", params, wtp_threshold=50000)
print(f"Value of Perspective: ${discordance['loss_from_discordance']:,.0f}")
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [API Reference](api/cea_model.md)

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17759272.svg)](https://doi.org/10.5281/zenodo.17759272)

- **Zenodo**: [10.5281/zenodo.17759272](https://doi.org/10.5281/zenodo.17759272)
- **OSF**: [osf.io/unq76](https://osf.io/unq76)

## License

Apache License 2.0
