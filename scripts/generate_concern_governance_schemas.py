#!/usr/bin/env python3
"""Generate deterministic concern-governance JSON Schemas."""

from __future__ import annotations

import argparse
from pathlib import Path

from vop_poc_nz.concerns import export_governance_schemas


def main() -> int:
    """Export schemas to the requested repository-relative directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schemas/governance"),
        help="schema output directory",
    )
    args = parser.parse_args()
    for path in export_governance_schemas(args.output):
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
