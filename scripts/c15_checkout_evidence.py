#!/usr/bin/env python3
"""Record and enforce the exact source commit exercised by a C15 job."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def checkout_evidence(
    repo: Path, *, expected_source_sha: str, event_name: str, runner: str
) -> dict[str, object]:
    """Return a fail-closed source/tested commit binding for one hosted job."""
    if len(expected_source_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_source_sha
    ):
        raise ValueError(
            "expected source SHA must be a lowercase 40-character hex digest"
        )
    tested_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return {
        "schema_version": "1.0.0",
        "event_name": event_name,
        "runner": runner,
        "expected_source_sha": expected_source_sha,
        "tested_sha": tested_sha,
        "exact_source_checkout": tested_sha == expected_source_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = checkout_evidence(
        args.repo.resolve(),
        expected_source_sha=args.expected_source_sha,
        event_name=args.event_name,
        runner=args.runner,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["exact_source_checkout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
