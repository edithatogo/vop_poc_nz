#!/usr/bin/env python3
"""Prevent the known Ruff debt from increasing while it is paid down."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

TARGETS = ("src/vop_poc_nz", "tests", "scripts")


def collect_counts(root: Path) -> dict[str, int]:
    """Return Ruff violation counts grouped by rule code."""
    completed = subprocess.run(
        ["ruff", "check", *TARGETS, "--output-format=json"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "ruff invocation failed")
    findings = json.loads(completed.stdout)
    return dict(sorted(Counter(item["code"] for item in findings).items()))


def ratchet_failures(current: dict[str, int], maximum: dict[str, int]) -> list[str]:
    """Report new rules and rule counts above the checked-in baseline."""
    return [
        f"{code}: {count} exceeds baseline {maximum.get(code, 0)}"
        for code, count in current.items()
        if count > maximum.get(code, 0)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--baseline", type=Path, default=Path("quality/ruff-baseline.json")
    )
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    baseline_path = (
        args.baseline if args.baseline.is_absolute() else root / args.baseline
    )
    current = collect_counts(root)
    if args.write_baseline:
        payload = {
            "schema_version": "1.0.0",
            "targets": list(TARGETS),
            "maximum_by_rule": current,
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(baseline_path)
        return 0
    maximum = json.loads(baseline_path.read_text(encoding="utf-8"))["maximum_by_rule"]
    failures = ratchet_failures(current, maximum)
    print(f"Ruff findings: {sum(current.values())}; baseline: {sum(maximum.values())}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
