"""Automation-contract tests for the bounded C13 governance harness."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from scripts.governance_harness import governance_commands

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_schema_regeneration_and_canonical_ledger_check_are_clean() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_concern_governance_schemas.py",
            "--check",
            "--ledger",
            "governance/registry.json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "governance schemas and ledger are current" in completed.stdout


def test_harness_commands_are_strict_focused_and_private_safe() -> None:
    commands = governance_commands(REPO_ROOT)
    rendered = "\n".join(" ".join(command.argv) for command in commands)

    assert [command.name for command in commands] == [
        "schema_and_ledger",
        "governance_tests",
        "import_boundary",
        "ruff",
        "ruff_format",
        "basedpyright",
    ]
    assert "tests/test_concerns.py" in rendered
    assert "tests/test_github_sync_planner.py" in rendered
    assert "scripts/import_boundary.py" in rendered
    assert "src/vop_poc_nz/concerns.py" in rendered
    assert "src/vop_poc_nz/github_sync_planner.py" in rendered
    assert ".conductor/local" not in rendered
    assert "pytest tests -q" not in rendered


def test_ci_pixi_and_mutation_profile_lanes_use_the_single_harness() -> None:
    workflow = (REPO_ROOT / ".github/workflows/quality-frontier.yml").read_text(
        encoding="utf-8"
    )
    pixi = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert workflow.count("scripts/governance_harness.py . --strict") == 1
    assert "scripts/profile_governance_workload.py" in workflow
    assert "github.event_name == 'schedule'" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert pixi["tasks"]["governance"]["cmd"] == (
        "uv run python scripts/governance_harness.py . --strict"
    )
    assert "governance" in pixi["tasks"]["verify"]["depends-on"]
    assert project["tool"]["mutmut"]["paths_to_mutate"] == [
        "src/vop_poc_nz/logging_config.py",
        "src/vop_poc_nz/github_sync_planner.py",
    ]
    assert "test_github_sync_planner.py" in project["tool"]["mutmut"]["runner"]
