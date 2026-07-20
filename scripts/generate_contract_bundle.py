#!/usr/bin/env python3
"""Generate or verify the canonical versioned VOP-VOIAGE contract bundle."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from vop_poc_nz.contract_bundle import BUNDLE_VERSION, export_contract_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contracts/vop-voiage") / BUNDLE_VERSION,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    schema_source = Path("schemas/domain")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="vop-contract-bundle-") as temp:
            generated = Path(temp) / BUNDLE_VERSION
            export_contract_bundle(generated, schema_source)
            expected = {
                path.relative_to(generated).as_posix(): path.read_bytes()
                for path in generated.rglob("*")
                if path.is_file()
            }
            actual = {
                path.relative_to(args.output).as_posix(): path.read_bytes()
                for path in args.output.rglob("*")
                if path.is_file()
            }
            if actual != expected:
                print("contract bundle is stale")
                return 2
        print("contract bundle is current")
        return 0
    manifest = export_contract_bundle(args.output, schema_source)
    print(manifest.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
