#!/usr/bin/env python3
"""Run the bounded C13 governance quality gates and record local evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GovernanceCommand:
    """One focused governance command."""

    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class GovernanceCommandResult:
    """Captured result for one command."""

    name: str
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def governance_commands(
    repo: Path, *, output_dir: Path | None = None
) -> tuple[GovernanceCommand, ...]:
    """Return the single canonical set of bounded governance checks."""
    root = repo.resolve()
    reports = output_dir or Path(tempfile.gettempdir()) / "vop-governance-harness"
    source_files = (
        "src/vop_poc_nz/concerns.py",
        "src/vop_poc_nz/compat/legacy.py",
        "src/vop_poc_nz/critical_invariants.py",
        "src/vop_poc_nz/domain/base.py",
        "src/vop_poc_nz/domain/cea.py",
        "src/vop_poc_nz/domain/contracts.py",
        "src/vop_poc_nz/github_sync_planner.py",
        "src/vop_poc_nz/kernels/base.py",
        "src/vop_poc_nz/kernels/cea.py",
        "src/vop_poc_nz/pipeline/typed.py",
        "src/vop_poc_nz/perspective_io.py",
        "src/vop_poc_nz/results/base.py",
        "src/vop_poc_nz/results/cea.py",
        "src/vop_poc_nz/results/pipeline.py",
        "src/vop_poc_nz/mutation_policy.py",
        "scripts/check_mutation_targets.py",
        "scripts/experimental_backend_probe.py",
        "scripts/generate_concern_governance_schemas.py",
        "scripts/generate_domain_contract_schemas.py",
        "scripts/governance_harness.py",
        "scripts/plan_github_governance_sync.py",
        "scripts/profile_governance_workload.py",
        "scripts/run_critical_mutation_lane.py",
    )
    test_files = (
        "tests/test_c13_contract_hardening.py",
        "tests/test_concerns.py",
        "tests/test_critical_invariants.py",
        "tests/test_critical_mutation_lane.py",
        "tests/test_experimental_backend_probe.py",
        "tests/test_github_sync_planner.py",
        "tests/test_governance_harness.py",
        "tests/test_import_boundary.py",
        "tests/test_mutation_score.py",
        "tests/test_perspective_io.py",
        "tests/test_typed_cea_contract.py",
        "tests/test_typed_pipeline.py",
    )
    lint_test_files = test_files[:-1]
    python = sys.executable
    return (
        GovernanceCommand(
            "schema_and_ledger",
            (
                python,
                "scripts/generate_concern_governance_schemas.py",
                "--check",
                "--ledger",
                "governance/registry.json",
            ),
        ),
        GovernanceCommand(
            "domain_schemas",
            (python, "scripts/generate_domain_contract_schemas.py", "--check"),
        ),
        GovernanceCommand(
            "governance_tests",
            (
                python,
                "-m",
                "pytest",
                *test_files,
                "-q",
                "--no-cov",
                "-p",
                "no:cacheprovider",
            ),
        ),
        GovernanceCommand(
            "import_boundary",
            (
                python,
                "scripts/import_boundary.py",
                str(root),
                "--strict",
                "--output-json",
                str(reports / "import_boundary.json"),
                "--output-md",
                str(reports / "import_boundary.md"),
            ),
        ),
        GovernanceCommand(
            "ruff", (python, "-m", "ruff", "check", *source_files, *lint_test_files)
        ),
        GovernanceCommand(
            "ruff_format",
            (
                python,
                "-m",
                "ruff",
                "format",
                "--check",
                *source_files,
                *lint_test_files,
            ),
        ),
        GovernanceCommand(
            "basedpyright",
            (python, "-m", "basedpyright", *source_files),
        ),
        GovernanceCommand("ty", (python, "-m", "ty", "check", *source_files)),
    )


def main() -> int:
    """Execute focused commands and write an untracked local JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    with tempfile.TemporaryDirectory(prefix="vop-governance-harness-") as temp:
        commands = governance_commands(repo, output_dir=Path(temp))
        results: list[GovernanceCommandResult] = []
        for command in commands:
            completed = subprocess.run(
                command.argv,
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )
            results.append(
                GovernanceCommandResult(
                    name=command.name,
                    argv=command.argv,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )

    report = {
        "schema_version": "1.0.0",
        "network_mutation": False,
        "private_content_published": False,
        "all_passed": all(result.returncode == 0 for result in results),
        "results": [asdict(result) for result in results],
    }
    output = repo / ".conductor/local/governance_harness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for result in results:
        print(f"{result.name}: {'passed' if result.returncode == 0 else 'failed'}")
    print(output)
    return 2 if args.strict and not report["all_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
