#!/usr/bin/env python3
"""Generate deterministic JSON Schemas for C13 typed public contracts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from vop_poc_nz.domain.cea import InterventionSpec
from vop_poc_nz.domain.contracts import (
    AnalysisSpec,
    DistributionSpec,
    NumericalPolicySpec,
    ParameterSpec,
    RunContextSpec,
)
from vop_poc_nz.results.base import ResultMetadata
from vop_poc_nz.results.cea import CEAAnalysisResult
from vop_poc_nz.results.pipeline import TypedPipelineResult

SCHEMAS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("analysis-spec", AnalysisSpec),
    ("cea-analysis-result", CEAAnalysisResult),
    ("distribution-spec", DistributionSpec),
    ("intervention-spec", InterventionSpec),
    ("numerical-policy-spec", NumericalPolicySpec),
    ("parameter-spec", ParameterSpec),
    ("result-metadata", ResultMetadata),
    ("run-context-spec", RunContextSpec),
    ("typed-pipeline-result", TypedPipelineResult),
)


def export_schemas(output: Path) -> tuple[Path, ...]:
    """Write stable, sorted schemas and return their paths."""
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, model in SCHEMAS:
        path = output / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        paths.append(path)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("schemas/domain"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="vop-domain-schema-") as temp:
            generated = export_schemas(Path(temp))
            expected = {path.name for path in generated}
            actual = {path.name for path in args.output.glob("*.json")}
            if expected != actual:
                print("domain schema file set is stale")
                return 2
            for generated_path in generated:
                committed = args.output / generated_path.name
                if generated_path.read_bytes() != committed.read_bytes():
                    print(f"domain schema is stale: {committed.as_posix()}")
                    return 2
        print("domain schemas are current")
        return 0
    for path in export_schemas(args.output):
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
