#!/usr/bin/env python3
"""Enforce per-target Mutmut score and unresolved-debt ratchets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vop_poc_nz.mutation_policy import (
    mutation_score_from_meta,
    mutation_target_report,
)


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"baseline {field} must be a non-negative integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("mutants"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    targets = baseline.get("targets") if isinstance(baseline, dict) else None
    if not isinstance(targets, dict) or not targets:
        raise ValueError("mutation baseline requires a non-empty targets object")

    cache_root = args.cache_root.resolve()
    reports: dict[str, object] = {}
    for target, expected in sorted(targets.items()):
        if not isinstance(target, str) or not isinstance(expected, dict):
            raise ValueError("mutation baseline target entries are invalid")
        relative_meta = expected.get("cache_metadata")
        if not isinstance(relative_meta, str):
            raise ValueError(f"baseline target {target} requires cache_metadata")
        meta = (cache_root / relative_meta).resolve()
        if not meta.is_relative_to(cache_root):
            raise ValueError(f"baseline target {target} escapes the cache root")
        score = mutation_score_from_meta(meta)
        reports[target] = mutation_target_report(
            score,
            baseline_killed=_count(expected.get("killed"), "killed"),
            baseline_eligible=_count(expected.get("eligible"), "eligible"),
        )

    passed = all(bool(report["passed"]) for report in reports.values())  # type: ignore[index]
    rendered = (
        json.dumps(
            {"schema_version": "1.0.0", "targets": reports, "passed": passed},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
