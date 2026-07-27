#!/usr/bin/env python3
"""Validate C16 and optionally dispatch its bounded consumer notification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vop_poc_nz.specialized_voi_projection import (
    dispatch,
    dispatch_plan,
    load_projection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path("conductor/tracks/specialized-voi-v1-2_20260727/projection.json"),
    )
    parser.add_argument("--canonical-ref", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    plan = dispatch_plan(load_projection(args.projection), args.canonical_ref)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    if args.dispatch:
        dispatch(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
