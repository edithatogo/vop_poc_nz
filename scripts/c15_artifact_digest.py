#!/usr/bin/env python3
"""Record or compare cross-platform normalized artifact digests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from vop_poc_nz.c15_reproducibility import (
    ArtifactMismatch,
    compare_digest_reports,
    normalized_archive_report,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"digest report must be an object: {path}")
    return cast(dict[str, object], payload)


def _single_archive(directory: Path, *, kind: str | None = None) -> Path:
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.suffix in {".whl", ".zip"} or path.name.endswith(".tar.gz"))
        and (
            kind is None
            or (kind == "wheel" and path.suffix == ".whl")
            or (kind == "sdist" and path.name.endswith(".tar.gz"))
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one supported archive in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--archive", type=Path)
    record.add_argument("--archive-dir", type=Path)
    record.add_argument("--kind", choices=["wheel", "sdist"])
    record.add_argument("--runner", required=True)
    record.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "record":
            if (args.archive is None) == (args.archive_dir is None):
                raise ValueError("provide exactly one of --archive or --archive-dir")
            archive = args.archive or _single_archive(args.archive_dir, kind=args.kind)
            payload = normalized_archive_report(archive, runner=args.runner)
        else:
            payload = compare_digest_reports(_load(args.left), _load(args.right))
        _write(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (ArtifactMismatch, OSError, ValueError) as exc:
        failure: dict[str, object] = {
            "schema_version": "1.0.0",
            "operation": args.command,
            "status": "failure",
            "matched": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
