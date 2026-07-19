from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
    REPO_ROOT / "scripts",
    REPO_ROOT / "examples",
)
SHIM_MODULES = (
    "bia_model",
    "cea_model_core",
    "cluster_analysis",
    "dcea_equity_analysis",
    "discordance_analysis",
    "dsa_analysis",
    "logging_config",
    "main",
    "main_analysis",
    "perspective_value_dsa",
    "perspective_value_viz",
    "policy_brief_generator",
    "profile_scalability",
    "profiling",
    "reporting",
    "sobol_analysis",
    "test_dsa_enhancements",
    "threshold_analysis",
    "validation",
    "value_of_information",
    "visualizations",
    "visualizations_comparative",
    "visualizations_extended",
)


def _source_tree_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            findings.extend(
                (node.lineno, alias.name)
                for alias in node.names
                if alias.name == "src" or alias.name.startswith("src.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "src" or node.module.startswith("src."):
                findings.append((node.lineno, node.module))
    return findings


def test_tests_and_scripts_do_not_import_source_tree_modules() -> None:
    paths = [*REPO_ROOT.glob("*.py")]
    paths.extend(path for root in IMPORT_ROOTS for path in root.rglob("*.py"))
    findings = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{line}: {module}"
        for path in paths
        for line, module in _source_tree_imports(path)
    ]
    assert findings == []


def test_legacy_module_shims_alias_canonical_modules() -> None:
    code = f"""
import importlib
import warnings
for module_name in {SHIM_MODULES!r}:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        legacy = importlib.import_module(f'src.{{module_name}}')
    canonical = importlib.import_module(f'vop_poc_nz.{{module_name}}')
    assert legacy is canonical, module_name
    assert any(item.category is DeprecationWarning for item in caught), module_name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0, completed.stderr
