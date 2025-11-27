"""
Core Analysis Pipeline Module.

This module contains the logic for running the health economic analysis,
including CEA, DCEA, VOI, and DSA. It separates the "math" from the "reporting".
"""

import os
from pathlib import Path
from typing import Dict

import yaml


def load_parameters(filepath: str = "src/parameters.yaml") -> Dict:
    """Load parameters from a YAML file."""
    # Path relative to project root (assuming run from root)
    if os.path.exists(filepath):
        with open(filepath) as f:
            return yaml.safe_load(f)

    # Fallback for running from src/pipeline
    project_root = Path(__file__).parent.parent.parent
    full_path = project_root / filepath
    if full_path.exists():
        with open(full_path) as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find parameters file at {filepath} or {full_path}"
    )
