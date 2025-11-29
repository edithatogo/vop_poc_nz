# Installation

## From PyPI (Recommended)

```bash
pip install vop-poc-nz
```

## From Source

```bash
git clone https://github.com/edithatogo/vop_poc_nz.git
cd vop_poc_nz
pip install -e ".[dev]"
```

## Optional Dependencies

### Visualization Extras

For enhanced plotting with plotnine and scienceplots:

```bash
pip install vop-poc-nz[viz]
```

### Reporting Extras

For Excel export and Pydantic models:

```bash
pip install vop-poc-nz[reporting]
```

### Development

For development tools (pytest, ruff, mypy, etc.):

```bash
pip install vop-poc-nz[dev]
```

## Requirements

- Python 3.9+
- NumPy, Pandas, SciPy, Matplotlib
- See [pyproject.toml](https://github.com/edithatogo/vop_poc_nz/blob/main/pyproject.toml) for full list

## Verify Installation

```python
import vop_poc_nz
print(vop_poc_nz.__version__)
```
