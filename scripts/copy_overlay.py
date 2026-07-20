#!/usr/bin/env python3
"""Copy a repo overlay into a target checkout.

The script refuses to overwrite existing files unless `--force` is supplied.
Use `--dry-run` first in coding-agent workflows.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    overlay = args.overlay.resolve()
    target = args.target.resolve()
    if not overlay.exists():
        raise SystemExit(f"overlay does not exist: {overlay}")
    if not target.exists():
        raise SystemExit(f"target does not exist: {target}")

    planned: list[tuple[Path, Path]] = []
    conflicts: list[Path] = []
    for src in iter_files(overlay):
        rel = src.relative_to(overlay)
        dst = target / rel
        planned.append((src, dst))
        if dst.exists() and not args.force:
            conflicts.append(dst)

    if conflicts:
        print("Refusing to overwrite existing files:")
        for path in conflicts:
            print(f"  {path}")
        print("Rerun with --force only after reviewing diffs.")
        return 2

    for src, dst in planned:
        print(f"{'would copy' if args.dry_run else 'copy'} {src.relative_to(overlay)}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
