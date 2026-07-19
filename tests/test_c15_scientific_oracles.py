from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vop_poc_nz.c15_scientific_oracles import decimal_evpi, numpy_evpi
from vop_poc_nz.value_of_information import calculate_evpi

FIXTURE = Path(__file__).parent / "fixtures/c15_scientific_oracles.json"


def _cases() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["case_id"]))
def test_literal_oracles_match_decimal_and_numpy(case: dict[str, object]) -> None:
    expected = Decimal(str(case["expected_evpi"]))
    matrix = case["net_benefit"]
    assert decimal_evpi(matrix) == expected
    assert numpy_evpi(np.asarray(matrix, dtype=np.float64)) == pytest.approx(
        float(expected), rel=1e-12, abs=1e-15
    )


@pytest.mark.parametrize(
    "case", [case for case in _cases() if len(case["net_benefit"][0]) == 2]
)
def test_two_strategy_oracles_match_production_pandas_backend(
    case: dict[str, object],
) -> None:
    matrix = np.asarray(case["net_benefit"], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "qaly_sc": matrix[:, 0],
            "cost_sc": np.zeros(matrix.shape[0]),
            "qaly_nt": matrix[:, 1],
            "cost_nt": np.zeros(matrix.shape[0]),
        }
    )
    actual = calculate_evpi(frame, wtp_threshold=1.0)
    assert actual == pytest.approx(
        float(case["production_expected_evpi"]), rel=1e-12, abs=1e-15
    )
    assert abs(actual - float(case["expected_evpi"])) <= float(
        case["production_zero_tolerance"]
    )


def test_higher_dimensional_oracle_matches_polars_backend() -> None:
    pl = pytest.importorskip("polars")
    case = next(item for item in _cases() if item["case_id"] == "three-strategy")
    frame = pl.DataFrame(case["net_benefit"], orient="row")
    per_draw = frame.max_horizontal().mean()
    current = max(frame[column].mean() for column in frame.columns)
    assert per_draw - current == pytest.approx(float(case["expected_evpi"]))


def test_higher_dimensional_oracle_matches_optional_jax_backend() -> None:
    jnp = pytest.importorskip("jax.numpy")
    case = next(item for item in _cases() if item["case_id"] == "three-strategy")
    values = jnp.asarray(case["net_benefit"], dtype=jnp.float32)
    actual = jnp.mean(jnp.max(values, axis=1)) - jnp.max(jnp.mean(values, axis=0))
    assert float(actual) == pytest.approx(float(case["expected_evpi"]), abs=1e-6)
