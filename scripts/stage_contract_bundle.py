#!/usr/bin/env python3
"""Stage a deterministic standalone VOP-VOIAGE contract bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vop_poc_nz.contract_bundle_staging import stage_contract_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--tag-object-oid", required=True)
    parser.add_argument("--tag-target-commit", required=True)
    parser.add_argument("--tag-verification", type=Path, required=True)
    args = parser.parse_args()
    stage = stage_contract_bundle(
        args.bundle,
        args.output,
        release_tag=args.release_tag,
        source_revision=args.source_revision,
        expected_bundle_sha256=args.bundle_sha256,
        tag_object_oid=args.tag_object_oid,
        tag_target_commit=args.tag_target_commit,
        tag_verification_path=args.tag_verification,
    )
    print(json.dumps(stage, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
