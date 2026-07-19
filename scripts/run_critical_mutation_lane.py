#!/usr/bin/env python3
"""Run a self-contained strict mutation lane over production C13 invariants."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_CONFIG = """\
[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.mutmut]
source_paths = ["src/vop_poc_nz/"]
only_mutate = ["src/vop_poc_nz/critical_invariants.py"]
pytest_add_cli_args = ["-x", "--no-cov", "-p", "no:cacheprovider"]
pytest_add_cli_args_test_selection = ["tests/test_critical_invariants.py"]
"""


def _non_negative_int(raw: dict[str, object], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"mutation statistic {field} must be a non-negative integer")
    return value


def mutation_report(raw: dict[str, object], *, threshold: float) -> dict[str, object]:
    """Score every non-skipped mutant, including no-test and timeout outcomes."""
    if not 0.0 < threshold <= 100.0:
        raise ValueError("threshold must be greater than 0 and at most 100")
    counts = {
        field: _non_negative_int(raw, field)
        for field in (
            "killed",
            "survived",
            "no_tests",
            "suspicious",
            "timeout",
            "segfault",
            "skipped",
            "total",
        )
    }
    interrupted = _non_negative_int(raw, "check_was_interrupted_by_user")
    accounted = (
        counts["killed"]
        + counts["survived"]
        + counts["no_tests"]
        + counts["suspicious"]
        + counts["timeout"]
        + counts["segfault"]
        + counts["skipped"]
        + interrupted
    )
    if counts["total"] < accounted:
        raise ValueError("mutation total is smaller than reported status counts")
    eligible = counts["total"] - counts["skipped"]
    score = 0.0 if eligible <= 0 else 100.0 * counts["killed"] / eligible
    passed = interrupted == 0 and eligible > 0 and score >= threshold
    return {
        **counts,
        "interrupted": interrupted,
        "eligible": eligible,
        "score_percent": round(score, 3),
        "threshold_percent": threshold,
        "passed": passed,
    }


def _run(repo: Path, output: Path, threshold: float) -> int:
    with tempfile.TemporaryDirectory(prefix="vop-critical-mutation-") as temp:
        sandbox = Path(temp)
        package = sandbox / "src/vop_poc_nz"
        tests = sandbox / "tests"
        package.mkdir(parents=True)
        tests.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8", newline="\n")
        shutil.copy2(
            repo / "src/vop_poc_nz/critical_invariants.py",
            package / "critical_invariants.py",
        )
        shutil.copy2(repo / "tests/test_critical_invariants.py", tests)
        (sandbox / "pyproject.toml").write_text(_CONFIG, encoding="utf-8", newline="\n")
        subprocess.run([sys.executable, "-m", "mutmut", "run"], cwd=sandbox, check=True)
        subprocess.run(
            [sys.executable, "-m", "mutmut", "export-cicd-stats"],
            cwd=sandbox,
            check=True,
        )
        stats = json.loads(
            (sandbox / "mutants/mutmut-cicd-stats.json").read_text(encoding="utf-8")
        )
    if not isinstance(stats, dict):
        raise TypeError("Mutmut statistics must be a JSON object")
    report = mutation_report(stats, threshold=threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/mutation-critical.json"),
    )
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()
    return _run(args.repo.resolve(), args.output.resolve(), args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
