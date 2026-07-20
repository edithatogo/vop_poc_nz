#!/usr/bin/env python3
"""Initialise and update resumable local conductor track state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = {"planned", "ready", "in_progress", "blocked", "completed", "superseded"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(pack_root: Path) -> dict[str, Any]:
    return json.loads((pack_root / "conductor" / "manifest.json").read_text(encoding="utf-8"))


def state_path(repo: Path) -> Path:
    return repo / ".conductor" / "local" / "track_state.json"


def initialise(repo: Path, pack_root: Path, *, force: bool = False) -> dict[str, Any]:
    path = state_path(repo)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = load_manifest(pack_root)
    state = {
        "schema_version": "1.0",
        "pack_version": manifest.get("pack_version"),
        "repo": str(repo.resolve()),
        "updated_at": now(),
        "tracks": {
            track["id"]: {
                "status": track.get("default_status", "planned"),
                "notes": [],
                "evidence": [],
                "commits": [],
                "updated_at": now(),
            }
            for track in manifest.get("tracks", [])
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def update(repo: Path, pack_root: Path, track_id: str, status: str, *, note: str | None, evidence: list[str], commits: list[str]) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Invalid status: {status}")
    state = initialise(repo, pack_root)
    if track_id not in state["tracks"]:
        raise ValueError(f"Unknown track: {track_id}")
    entry = state["tracks"][track_id]
    if status == "completed" and not (evidence or entry.get("evidence")):
        raise ValueError("Completing a track requires at least one evidence reference")
    entry["status"] = status
    if note:
        entry.setdefault("notes", []).append({"at": now(), "text": note})
    entry.setdefault("evidence", []).extend(item for item in evidence if item not in entry.get("evidence", []))
    entry.setdefault("commits", []).extend(item for item in commits if item not in entry.get("commits", []))
    entry["updated_at"] = now()
    state["updated_at"] = now()
    state_path(repo).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def to_markdown(state: dict[str, Any]) -> str:
    lines = ["# Conductor track state", "", "| Track | Status | Evidence | Commits |", "|---|---|---:|---:|"]
    for track_id, entry in sorted(state["tracks"].items()):
        lines.append(f"| `{track_id}` | {entry['status']} | {len(entry.get('evidence', []))} | {len(entry.get('commits', []))} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--track")
    parser.add_argument("--status", choices=sorted(STATUSES))
    parser.add_argument("--note")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--commit", dest="commits", action="append", default=[])
    args = parser.parse_args()
    repo = args.repo.resolve()
    pack_root = args.pack_root.resolve()
    if args.track or args.status:
        if not args.track or not args.status:
            parser.error("--track and --status must be supplied together")
        state = update(repo, pack_root, args.track, args.status, note=args.note, evidence=args.evidence, commits=args.commits)
    else:
        state = initialise(repo, pack_root, force=args.force)
    md_path = repo / ".conductor" / "local" / "track_state.md"
    md_path.write_text(to_markdown(state), encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
