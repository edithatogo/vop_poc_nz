#!/usr/bin/env python3
"""Copy only pack-doctor-approved missing files; never overwrite or merge.

Mutating integration is guarded by ``git_safety``. By default it refuses to
write on a default branch, detached HEAD, or a worktree with tracked/staged
changes. Overrides are explicit and recorded in the local integration audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import git_safety
import pack_doctor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--pack-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-default-branch", action="store_true")
    parser.add_argument("--allow-detached", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = pack_doctor.build_report(repo, args.pack_root)
    safety = git_safety.inspect(repo)
    copied: list[str] = []
    if args.apply:
        git_safety.require_safe(
            repo,
            allow_dirty=args.allow_dirty,
            allow_default_branch=args.allow_default_branch,
            allow_detached=args.allow_detached,
        )
        for item in report["items"]:
            if item["status"] != "safe_add":
                continue
            source = Path(item["source"])
            target = Path(item["target"])
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(repo)))
    record = {
        "schema_version": "1.1",
        "at": datetime.now(timezone.utc).isoformat(),
        "applied": args.apply,
        "copied": copied,
        "skipped": len(report["items"]) - len(copied),
        "git_safety": safety.as_dict(),
        "overrides": {
            "allow_dirty": args.allow_dirty,
            "allow_default_branch": args.allow_default_branch,
            "allow_detached": args.allow_detached,
        },
    }
    out = repo / ".conductor" / "local" / "safe_integration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Safe additions copied: {len(copied)}" if args.apply else f"Dry run: {report['summary'].get('safe_add', 0)} safe additions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
