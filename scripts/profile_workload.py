"""Deterministic Arrow/Polars workload used by Scalene and CI."""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import polars as pl
import pyarrow as pa


def main() -> None:
    rng = np.random.default_rng(20260719)
    draws = rng.normal(size=(100_000, 4))
    table = pa.table({f"strategy_{index}": draws[:, index] for index in range(4)})
    frame = cast("pl.DataFrame", pl.from_arrow(table))
    summary = frame.select(pl.all().mean()).to_dicts()[0]
    print(json.dumps({"rows": table.num_rows, "means": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
