# Makefile for health economics canonical code

.PHONY: help install test lint format check clean

# Default target
help:
	@echo "Health Economics Canonical Code - Development Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install the package in development mode with dev dependencies"
	@echo "  make test         Run all tests with pytest"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make lint         Lint code with ruff"
	@echo "  make format       Format code with black"
	@echo "  make check        Run linting, formatting, and tests"
	@echo "  make clean        Clean up build artifacts and cache files"
	@echo ""

# Install the package in development mode
install:
	pip install -e .[dev]

# Run tests
test:
	pytest tests/

# Run tests with coverage
test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Run linting with ruff
lint:
	ruff check src/ tests/

# Format code with black
format:
	black src/ tests/

# Run all checks (linting, formatting, tests)
check: lint test

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf src/*.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete