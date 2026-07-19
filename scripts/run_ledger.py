#!/usr/bin/env python3
"""Append-only local ledger for conductor prompt/agent runs.

The ledger is intentionally stored under `.conductor/local/` so it can include
local paths, prompt IDs, and decision notes without being pushed. It gives a
coding agent continuity across a prompt series: which repo map was used, what was
applied, which files were touched, and what remains local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def record_event(
    repo: Path,
    *,
    prompt_id: str,
    agent: str,
    status: str,
    action: str,
    notes: str = "",
    touched_paths: list[str] | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    local_dir = repo / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    repo_map_path = local_dir / "repo_map.json"
    integration_plan_path = local_dir / "integration_plan.json"
    record = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "root": str(repo),
            "name": repo.name,
            "branch": git_value(repo, ["branch", "--show-current"]),
            "head": git_value(repo, ["rev-parse", "--short", "HEAD"]),
            "dirty_status_hash": sha256_text(git_value(repo, ["status", "--short"]) or ""),
        },
        "repo_map_sha256": sha256_file(repo_map_path),
        "integration_plan_sha256": sha256_file(integration_plan_path),
        "prompt_id": prompt_id,
        "agent": agent,
        "status": status,
        "action": action,
        "notes": notes,
        "touched_paths": touched_paths or [],
    }
    ledger = local_dir / "run_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def read_events(repo: Path) -> list[dict[str, object]]:
    ledger = repo / ".conductor" / "local" / "run_ledger.jsonl"
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_markdown(repo: Path) -> str:
    events = read_events(repo)
    lines = [f"# Local conductor run ledger: {repo.name}", "", f"- Events: {len(events)}", ""]
    lines.extend(["| Time | Prompt | Agent | Status | Action | Touched paths |", "|---|---|---|---|---|---|"])
    for event in events[-100:]:
        paths = ", ".join(f"`{p}`" for p in event.get("touched_paths", [])[:8])
        if len(event.get("touched_paths", [])) > 8:
            paths += ", ..."
        lines.append(
            f"| {event['timestamp_utc']} | `{event['prompt_id']}` | `{event['agent']}` | `{event['status']}` | {event['action']} | {paths} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="Append a run event")
    record.add_argument("repo", type=Path)
    record.add_argument("--prompt-id", required=True)
    record.add_argument("--agent", default="local-agent")
    record.add_argument("--status", default="completed")
    record.add_argument("--action", required=True)
    record.add_argument("--notes", default="")
    record.add_argument("--touched-path", action="append", default=[])

    summary = subparsers.add_parser("summary", help="Write a Markdown summary")
    summary.add_argument("repo", type=Path)
    summary.add_argument("--output-md", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "record":
        event = record_event(
            args.repo,
            prompt_id=args.prompt_id,
            agent=args.agent,
            status=args.status,
            action=args.action,
            notes=args.notes,
            touched_paths=args.touched_path,
        )
        print(json.dumps(event, indent=2, sort_keys=True))
        return 0
    if args.command == "summary":
        text = summary_markdown(args.repo.resolve())
        output = args.output_md or args.repo.resolve() / ".conductor" / "local" / "run_ledger.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Run ledger summary written: {output}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
