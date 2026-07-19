#!/usr/bin/env python3
"""Fail if local-only/generated files are staged or tracked for publication.

This gate is intended to run before `git push`, release, PyPI/conda publication,
or manuscript-asset export. It is conservative: files classified as local-only or
generated fail; files that need review produce warnings unless --strict is used.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import repo_map
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Could not import repo_map.py from {SCRIPT_DIR}: {exc}") from exc


def run_git(root: Path, args: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return [line for line in completed.stdout.splitlines() if line]


def load_policy(root: Path, path: Path | None) -> dict[str, object]:
    candidate = path or root / ".conductor" / "publication_policy.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {
        "allow_tracked_globs": [],
        "deny_tracked_globs": [
            ".env*",
            ".conductor/local/**",
            "local/**",
            "private/**",
            "data/raw/**",
            "data/private/**",
            "artifacts/local/**",
            "outputs/local/**",
            "results/local/**",
            "*.pkl",
            "*.pickle",
            "*.sqlite",
            "*.sqlite3",
            "*.parquet",
        ],
    }


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail also on review_before_commit files")
    args = parser.parse_args()

    root = args.repo.resolve()
    policy = load_policy(root, args.policy)
    allow_patterns = list(policy.get("allow_tracked_globs", []))
    deny_patterns = list(policy.get("deny_tracked_globs", []))
    tracked = run_git(root, ["ls-files"])

    failures: list[str] = []
    warnings: list[str] = []
    for path in tracked:
        if matches_any(path, allow_patterns):
            continue
        category, publish_policy, reasons = repo_map.classify(path)
        reason = "; ".join(reasons)
        if matches_any(path, deny_patterns) or publish_policy == "do_not_commit":
            failures.append(f"{path} :: {category}/{publish_policy} :: {reason}")
        elif publish_policy == "review_before_commit":
            message = f"{path} :: {category}/{publish_policy} :: {reason}"
            if args.strict:
                failures.append(message)
            else:
                warnings.append(message)

    if warnings:
        print("Publication gate warnings:")
        for item in warnings:
            print(f"  WARN {item}")
    if failures:
        print("Publication gate failures:")
        for item in failures:
            print(f"  FAIL {item}")
        print("\nResolve by moving files to local workspaces, adding explicit allow rules, or untracking generated artifacts.")
        return 2
    print("Publication gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
