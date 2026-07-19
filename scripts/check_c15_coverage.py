#!/usr/bin/env python3
"""Enforce aggregate, critical-module, and changed-line branch coverage."""

# pyright: reportAny=false

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import cast

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", value)


def changed_python_lines(repo: Path, base: str) -> dict[str, set[int]]:
    """Return added or modified production Python lines from a zero-context diff."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            f"{base}...HEAD",
            "--",
            "src/vop_poc_nz",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    changed: dict[str, set[int]] = {}
    current: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            candidate = line[6:].replace("\\", "/")
            current = candidate if candidate.endswith(".py") else None
            if current is not None:
                changed.setdefault(current, set())
            continue
        match = HUNK.match(line)
        if current is not None and match:
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            changed[current].update(range(start, start + count))
    return {path: lines for path, lines in changed.items() if lines}


def evaluate_coverage(
    coverage: dict[str, object],
    policy: dict[str, object],
    changed: dict[str, set[int]],
) -> dict[str, object]:
    """Evaluate coverage.py JSON against all policy surfaces."""
    totals = cast("dict[str, object]", coverage["totals"])
    raw_files = cast("dict[str, dict[str, object]]", coverage["files"])
    files = {path.replace("\\", "/"): value for path, value in raw_files.items()}
    aggregate = float(cast("int | float", totals["percent_covered"]))
    aggregate_min = float(cast("int | float", policy["aggregate_percent"]))
    critical: list[dict[str, object]] = []
    for path, minimum in cast(
        "dict[str, int | float]", policy["critical_modules"]
    ).items():
        details = files.get(path)
        if details is None:
            critical.append(
                {
                    "path": path,
                    "minimum": float(minimum),
                    "missing": True,
                    "passed": False,
                }
            )
            continue
        summary = cast("dict[str, object]", details["summary"])
        percent = float(cast("int | float", summary["percent_covered"]))
        critical.append(
            {
                "path": path,
                "minimum": float(minimum),
                "percent": percent,
                "passed": percent >= float(minimum),
            }
        )

    unmeasured = sorted(path for path in changed if path not in files)
    executable = covered = branches = covered_branches = 0
    missing_lines: list[str] = []
    missing_branches: list[str] = []
    for path, lines in sorted(changed.items()):
        details = files.get(path)
        if details is None:
            continue
        executed_lines = set(cast("list[int]", details.get("executed_lines", [])))
        absent_lines = set(cast("list[int]", details.get("missing_lines", [])))
        relevant = lines & (executed_lines | absent_lines)
        executable += len(relevant)
        covered += len(relevant & executed_lines)
        missing_lines.extend(
            f"{path}:{line}" for line in sorted(relevant & absent_lines)
        )
        executed_pairs = {
            tuple(pair)
            for pair in cast("list[list[int]]", details.get("executed_branches", []))
        }
        missing_pairs = {
            tuple(pair)
            for pair in cast("list[list[int]]", details.get("missing_branches", []))
        }
        relevant_pairs = {
            pair for pair in executed_pairs | missing_pairs if pair[0] in lines
        }
        branches += len(relevant_pairs)
        covered_branches += len(relevant_pairs & executed_pairs)
        missing_branches.extend(
            f"{path}:{origin}->{destination}"
            for origin, destination in sorted(relevant_pairs & missing_pairs)
        )
    line_percent = 100.0 if executable == 0 else covered * 100 / executable
    branch_percent = 100.0 if branches == 0 else covered_branches * 100 / branches
    line_min = float(cast("int | float", policy["changed_line_percent"]))
    branch_min = float(cast("int | float", policy["changed_branch_percent"]))
    passed = (
        aggregate >= aggregate_min
        and all(bool(item["passed"]) for item in critical)
        and not unmeasured
        and line_percent >= line_min
        and branch_percent >= branch_min
    )
    return {
        "schema_version": "1.0.0",
        "aggregate": {"percent": aggregate, "minimum": aggregate_min},
        "critical_modules": critical,
        "changed": {
            "executable_lines": executable,
            "covered_lines": covered,
            "line_percent": line_percent,
            "line_minimum": line_min,
            "branches": branches,
            "covered_branches": covered_branches,
            "branch_percent": branch_percent,
            "branch_minimum": branch_min,
            "missing_lines": missing_lines,
            "missing_branches": missing_branches,
            "unmeasured_files": unmeasured,
        },
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, default=Path(".github/c15-coverage-policy.json")
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = evaluate_coverage(
        _object(args.coverage),
        _object(args.policy),
        changed_python_lines(repo, args.base),
    )
    report["base"] = args.base
    report["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
