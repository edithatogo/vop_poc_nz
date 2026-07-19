"""Automation-contract tests for the bounded C13 governance harness."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.generate_concern_governance_schemas import (
    validate_local_evidence_provenance,
)
from scripts.governance_harness import governance_commands
from vop_poc_nz.concerns import EvidenceReference, GovernanceLedger

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


def test_local_evidence_provenance_is_bound_to_pinned_git_bytes() -> None:
    ledger = GovernanceLedger.model_validate_json(
        (REPO_ROOT / "governance/registry.json").read_text(encoding="utf-8")
    )
    validate_local_evidence_provenance(ledger, REPO_ROOT)

    evidence = next(
        record for record in ledger.records if isinstance(record, EvidenceReference)
    )
    invalid = evidence.model_copy(update={"sha256": sha256(b"wrong").hexdigest()})
    invalid_ledger = GovernanceLedger(
        records=tuple(
            invalid if record.id == evidence.id else record for record in ledger.records
        )
    )
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        validate_local_evidence_provenance(invalid_ledger, REPO_ROOT)


def test_harness_commands_are_strict_focused_and_private_safe() -> None:
    commands = governance_commands(REPO_ROOT)
    rendered = "\n".join(" ".join(command.argv) for command in commands)

    assert [command.name for command in commands] == [
        "schema_and_ledger",
        "domain_schemas",
        "governance_tests",
        "import_boundary",
        "ruff",
        "ruff_format",
        "basedpyright",
        "ty",
    ]
    assert "tests/test_concerns.py" in rendered
    assert "scripts/generate_domain_contract_schemas.py --check" in rendered
    assert "tests/test_github_sync_planner.py" in rendered
    assert "scripts/import_boundary.py" in rendered
    assert "src/vop_poc_nz/concerns.py" in rendered
    assert "src/vop_poc_nz/critical_invariants.py" in rendered
    assert "src/vop_poc_nz/github_sync_planner.py" in rendered
    assert "src/vop_poc_nz/mutation_policy.py" in rendered
    assert "src/vop_poc_nz/perspective_io.py" in rendered
    assert "src/vop_poc_nz/domain/contracts.py" in rendered
    assert "src/vop_poc_nz/pipeline/typed.py" in rendered
    assert "tests/test_typed_cea_contract.py" in rendered
    assert "tests/test_typed_pipeline.py" in rendered
    assert "tests/test_c13_contract_hardening.py" in rendered
    assert "tests/test_critical_invariants.py" in rendered
    assert "tests/test_critical_mutation_lane.py" in rendered
    assert "tests/test_mutation_score.py" in rendered
    assert "tests/test_perspective_io.py" in rendered
    assert "scripts/check_mutation_targets.py" in rendered
    assert "scripts/experimental_backend_probe.py" in rendered
    assert "scripts/run_critical_mutation_lane.py" in rendered
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
    assert ".benchmarks/bandit.json" in workflow
    assert ".benchmarks/pip-audit.json" in workflow
    assert ".benchmarks/experimental-backend.json" in workflow
    assert pixi["tasks"]["governance"]["cmd"] == (
        "uv run python scripts/governance_harness.py . --strict"
    )
    assert "governance" in pixi["tasks"]["verify"]["depends-on"]
    assert project["tool"]["mutmut"]["source_paths"] == ["src/vop_poc_nz/"]
    assert project["tool"]["mutmut"]["only_mutate"] == [
        "src/vop_poc_nz/logging_config.py",
        "src/vop_poc_nz/github_sync_planner.py",
    ]
    baseline = json.loads(
        (REPO_ROOT / ".github/mutation-baselines/vop-broad.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(baseline["targets"]) == set(project["tool"]["mutmut"]["only_mutate"])
    mutation_tests = project["tool"]["mutmut"]["pytest_add_cli_args_test_selection"]
    assert mutation_tests == [
        "tests/test_logging_and_version.py",
        "tests/test_github_sync_planner.py",
        "tests/test_mutation_score.py",
    ]
    assert "mutmut export-cicd-stats" in workflow
    assert "scripts/check_mutation_targets.py" in workflow
    assert "scripts/run_critical_mutation_lane.py ." in workflow
    assert "--threshold 90" in workflow
    assert "continue-on-error: true" in workflow
    assert "steps.broad-mutation.outcome" in workflow
    assert "steps.critical-mutation.outcome" in workflow
    assert "include-hidden-files: true" in workflow
    assert "mutation-score" in pixi["tasks"]
