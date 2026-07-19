#!/usr/bin/env python3
"""Regenerate the small, public Arrow interoperability fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from vop_poc_nz.perspective_io import (
    read_ipc_records,
    schema_fingerprint,
    write_ipc_records,
    write_records,
)

RECORDS = [
    {
        "case_id": "fixture-v1",
        "draw": "0",
        "strategy": "usual-care",
        "perspective": "health-system",
        "net_benefit": 1250.5,
        "event_time": "2026-01-01T00:00:00Z",
        "subgroup": None,
    },
    {
        "case_id": "fixture-v1",
        "draw": "1",
        "strategy": "intervention",
        "perspective": "societal",
        "net_benefit": -42.25,
        "event_time": "2026-01-02T00:00:00Z",
        "subgroup": "priority",
    },
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    fixture_dir = root / "tests" / "fixtures" / "interchange" / "v1"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    parquet = write_records(RECORDS, fixture_dir / "perspective.parquet")
    arrow = write_ipc_records(RECORDS, fixture_dir / "perspective.arrow")
    table = read_ipc_records(arrow)
    contract = {
        "contract_version": "1.0.0",
        "schema_fingerprint": schema_fingerprint(table.schema),
        "records": RECORDS,
        "artifacts": [parquet.name, arrow.name],
    }
    (fixture_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
