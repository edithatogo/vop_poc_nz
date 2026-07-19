"""Input/output helpers for perspective-regret outputs.

The functions prefer Arrow/Parquet for machine-readable outputs but degrade to
JSON Lines when optional dependencies are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .perspective import NetBenefitTensor


def tensor_to_long_records(tensor: NetBenefitTensor) -> list[dict[str, Any]]:
    """Convert a `NetBenefitTensor` to long-form records."""
    records: list[dict[str, Any]] = []
    draw_ids = tensor.draw_ids or tuple(str(i) for i in range(tensor.n_draws))
    for draw_idx, draw_id in enumerate(draw_ids):
        for strategy_idx, strategy in enumerate(tensor.strategies):
            for perspective_idx, perspective in enumerate(tensor.perspectives):
                records.append(
                    {
                        "case_id": tensor.case_id,
                        "draw": draw_id,
                        "strategy": strategy,
                        "perspective": perspective,
                        "net_benefit": float(
                            tensor.values[draw_idx, strategy_idx, perspective_idx]
                        ),
                    }
                )
    return records


def write_records(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    prefer_parquet: bool = True,
) -> Path:
    """Write records to Parquet if possible, otherwise JSON Lines.

    Args:
        records: Iterable of JSON-like row dictionaries.
        path: Output path. If the suffix is `.parquet`, Parquet is attempted.
        prefer_parquet: If true, `.jsonl` paths are upgraded to `.parquet` when
            pyarrow is available.
    """
    rows = [dict(row) for row in records]
    out = Path(path)
    if prefer_parquet and out.suffix not in {".parquet", ".jsonl"}:
        out = out.with_suffix(".parquet")

    if prefer_parquet or out.suffix == ".parquet":
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]

            if out.suffix != ".parquet":
                out = out.with_suffix(".parquet")
            table = pa.Table.from_pylist(rows)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out)
            return out
        except Exception:
            # Optional dependency or schema conversion failure. Fall back to
            # line-oriented JSON rather than pickle.
            if out.suffix == ".parquet":
                out = out.with_suffix(".jsonl")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def write_perspective_outputs(
    tensor: NetBenefitTensor,
    output_dir: str | Path,
    *,
    reference_perspective: str,
    population: float | None = None,
) -> dict[str, Path]:
    """Write standard perspective-analysis outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "net_benefit_tensor": write_records(
            tensor_to_long_records(tensor), out / "net_benefit_tensor.parquet"
        ),
        "regret_matrix": write_records(
            tensor.regret_matrix(population=population), out / "regret_matrix.parquet"
        ),
        "perspective_acceptability_frontier": write_records(
            (row.as_dict() for row in tensor.perspective_acceptability_frontier()),
            out / "perspective_acceptability_frontier.parquet",
        ),
        "mcda_features": write_records(
            tensor.mcda_feature_records(
                reference_perspective=reference_perspective, population=population
            ),
            out / "mcda_features.parquet",
        ),
    }
