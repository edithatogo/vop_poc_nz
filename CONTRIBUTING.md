# Contributing to vop_poc_nz

Thank you for your interest in contributing to the Distributional Cost-Effectiveness Analysis Framework! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list to avoid duplicates. When creating a bug report, include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Python version and OS
- Relevant package versions (`pip freeze`)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- A clear description of the proposed feature
- The motivation/use case
- Example code or API design if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`tox`)
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/vop_poc_nz.git
cd vop_poc_nz

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Development Guidelines

### Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 88 characters
- Use type hints for all function signatures
- Write docstrings for all public functions/classes (Google style)

### Testing

- Write tests for all new functionality
- Maintain >95% test coverage
- Use pytest for testing
- Use Hypothesis for property-based tests where appropriate

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vop_poc_nz --cov-report=html

# Run specific test file
pytest tests/test_cea_model_core.py

# Run tox for full validation
tox
```

### Type Checking

- All code should pass mypy and pyright checks
- Use `from __future__ import annotations` for forward references

```bash
# Run type checks
mypy src/vop_poc_nz
pyright
```

### Documentation

- Update docstrings for any changed functionality
- Add examples to docstrings where helpful
- Update CHANGELOG.md for user-facing changes

## Release Process

Releases are automated via GitHub Actions:

1. Update version in `src/vop_poc_nz/__init__.py`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v0.x.x`
4. Push the tag: `git push origin v0.x.x`
5. Create a GitHub Release from the tag
6. CI will automatically publish to PyPI

## Questions?

Feel free to open an issue for any questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
