#!/usr/bin/env python3
"""Install or update the managed conductor block without overwriting local instructions.

Preview generation is read-only with respect to public source. ``--apply`` is a
mutating integration operation and therefore uses the same Git safety guard as
safe overlay copying.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import git_safety

BEGIN = "<!-- BEGIN VOP-CONDUCTOR MANAGED BLOCK -->"
END = "<!-- END VOP-CONDUCTOR MANAGED BLOCK -->"


def merge(existing: str, block: str) -> str:
    if BEGIN in existing and END in existing:
        start = existing.index(BEGIN)
        finish = existing.index(END, start) + len(END)
        return existing[:start].rstrip() + "\n\n" + block.strip() + "\n" + existing[finish:].lstrip("\n")
    if existing.strip():
        return existing.rstrip() + "\n\n" + block.strip() + "\n"
    return block.strip() + "\n"


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
    block = (args.pack_root / "integration" / "AGENTS_MANAGED_BLOCK.md").read_text(encoding="utf-8")
    canonical = repo / "AGENTS.md"
    lowercase = repo / "agents.md"
    existing = canonical.read_text(encoding="utf-8") if canonical.exists() else lowercase.read_text(encoding="utf-8") if lowercase.exists() else ""
    proposed = merge(existing, block)
    preview = repo / ".conductor" / "local" / "AGENTS.proposed.md"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(proposed, encoding="utf-8")
    if args.apply:
        git_safety.require_safe(
            repo,
            allow_dirty=args.allow_dirty,
            allow_default_branch=args.allow_default_branch,
            allow_detached=args.allow_detached,
        )
        canonical.write_text(proposed, encoding="utf-8")
        print(canonical)
        if lowercase.exists() and lowercase != canonical:
            print(f"Review and remove/rename legacy {lowercase.name} after confirming content was preserved.")
    else:
        print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
