#!/usr/bin/env python3
"""Generate deterministic concern-governance JSON Schemas."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from vop_poc_nz.concerns import GovernanceLedger, export_governance_schemas


def main() -> int:
    """Export schemas to the requested repository-relative directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schemas/governance"),
        help="schema output directory",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("governance/registry.json"),
        help="canonical ledger to validate",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed schemas differ from deterministic regeneration",
    )
    args = parser.parse_args()
    GovernanceLedger.model_validate_json(args.ledger.read_text(encoding="utf-8"))
    if args.check:
        with tempfile.TemporaryDirectory(prefix="vop-governance-schema-") as temp:
            generated = export_governance_schemas(temp)
            expected_names = {path.name for path in generated}
            actual_names = {path.name for path in args.output.glob("*.json")}
            if expected_names != actual_names:
                print("governance schema file set is stale")
                return 2
            for generated_path in generated:
                committed_path = args.output / generated_path.name
                if generated_path.read_bytes() != committed_path.read_bytes():
                    print(f"governance schema is stale: {committed_path.as_posix()}")
                    return 2
        print("governance schemas and ledger are current")
        return 0
    for path in export_governance_schemas(args.output):
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
