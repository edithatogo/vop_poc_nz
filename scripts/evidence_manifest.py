#!/usr/bin/env python3
"""Generate or verify deterministic evidence manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vop_poc_nz.evidence_manifest import (
    build_evidence_manifest,
    verify_evidence_manifest,
    write_evidence_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("manifest", type=Path)
    generate.add_argument("files", nargs="+", type=Path)
    generate.add_argument("--root", type=Path, default=Path.cwd())
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    if args.command == "generate":
        manifest = build_evidence_manifest(args.files, root=args.root)
        print(write_evidence_manifest(manifest, args.manifest))
        return 0

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = verify_evidence_manifest(payload, root=args.root)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
