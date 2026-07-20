#!/usr/bin/env python3
"""Render a dependency-aware conductor status dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(repo: Path, pack_root: Path) -> dict[str, Any]:
    manifest = load(pack_root / "conductor" / "manifest.json")
    state_path = repo / ".conductor" / "local" / "track_state.json"
    state = load(state_path) if state_path.exists() else {"tracks": {}}
    rows = []
    statuses = {tid: entry.get("status", "planned") for tid, entry in state.get("tracks", {}).items()}
    for track in manifest.get("tracks", []):
        tid = track["id"]
        deps = track.get("depends_on", [])
        unmet = [dep for dep in deps if statuses.get(dep) != "completed"]
        status = statuses.get(tid, track.get("default_status", "planned"))
        ready = status not in {"completed", "superseded"} and not unmet
        rows.append({"id": tid, "title": track["title"], "status": status, "depends_on": deps, "unmet_dependencies": unmet, "ready": ready})
    return {"schema_version": "1.0", "repo": str(repo), "summary": {"ready": sum(r["ready"] for r in rows), "completed": sum(r["status"] == "completed" for r in rows), "total": len(rows)}, "tracks": rows}


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Conductor status dashboard", "", f"Ready: **{report['summary']['ready']}** · Completed: **{report['summary']['completed']} / {report['summary']['total']}**", "", "| ID | Track | Status | Ready | Unmet dependencies |", "|---|---|---|---|---|"]
    for row in report["tracks"]:
        unmet = ", ".join(row["unmet_dependencies"]) or "—"
        lines.append(f"| `{row['id']}` | {row['title']} | {row['status']} | {'yes' if row['ready'] else 'no'} | {unmet} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = build(args.repo.resolve(), args.pack_root.resolve())
    out_dir = args.repo / ".conductor" / "local"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conductor_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "conductor_status.md").write_text(to_markdown(report), encoding="utf-8")
    print(out_dir / "conductor_status.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
