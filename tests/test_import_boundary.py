from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (REPO_ROOT / "tests", REPO_ROOT / "scripts")
SHIM_MODULES = (
    "cluster_analysis",
    "validation",
    "threshold_analysis",
    "policy_brief_generator",
    "profiling",
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
    findings = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{line}: {module}"
        for root in IMPORT_ROOTS
        for path in root.rglob("*.py")
        for line, module in _source_tree_imports(path)
    ]
    assert findings == []


@pytest.mark.parametrize("module_name", SHIM_MODULES)
def test_legacy_module_shims_alias_canonical_module(module_name: str) -> None:
    code = f"""
import importlib
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    legacy = importlib.import_module('src.{module_name}')
canonical = importlib.import_module('vop_poc_nz.{module_name}')
assert legacy is canonical
assert any(item.category is DeprecationWarning for item in caught)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
