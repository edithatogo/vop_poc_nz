#!/usr/bin/env python3
"""Generate deterministic concern-governance JSON Schemas."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path

from vop_poc_nz.concerns import (
    EvidenceReference,
    GovernanceLedger,
    export_governance_schemas,
)


def validate_local_evidence_provenance(
    ledger: GovernanceLedger, repository: Path
) -> None:
    """Bind verified local evidence to bytes at its declared Git commit."""
    root = repository.resolve()
    for record in ledger.records:
        if not (
            isinstance(record, EvidenceReference)
            and record.locator_kind == "local_path"
            and record.status == "verified"
        ):
            continue
        if record.git_commit is None or record.sha256 is None:
            raise ValueError(
                f"verified local evidence {record.id} requires git_commit and sha256"
            )
        locator = record.locator.replace("\\", "/")
        completed = subprocess.run(
            ["git", "show", f"{record.git_commit}:{locator}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"cannot resolve evidence {record.id} at {record.git_commit}: {detail}"
            )
        actual = sha256(completed.stdout).hexdigest()
        if actual != record.sha256:
            raise ValueError(
                f"evidence digest mismatch for {record.id}: "
                f"expected {record.sha256}, got {actual}"
            )


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
    ledger = GovernanceLedger.model_validate_json(
        args.ledger.read_text(encoding="utf-8")
    )
    validate_local_evidence_provenance(ledger, Path.cwd())
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
