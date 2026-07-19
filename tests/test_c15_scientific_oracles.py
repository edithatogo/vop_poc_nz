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


def _tolerance(case: dict[str, object], name: str) -> float:
    tolerances = case["tolerances"]
    assert isinstance(tolerances, dict)
    return float(tolerances[name])


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["case_id"]))
def test_oracle_case_declares_units_assumptions_and_tolerances(
    case: dict[str, object],
) -> None:
    assert case["units"] == {
        "net_benefit": "NZD_2025_per_person",
        "expected_evpi": "NZD_2025_per_person",
    }
    assert case["assumptions"]
    assert set(case["tolerances"]) == {
        "decimal_absolute",
        "numpy_absolute",
        "numpy_relative",
        "production_absolute",
        "production_relative",
        "jax_absolute",
    }


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["case_id"]))
def test_literal_oracles_match_decimal_and_numpy(case: dict[str, object]) -> None:
    expected = Decimal(str(case["expected_evpi"]))
    matrix = case["net_benefit"]
    assert decimal_evpi(matrix) == expected
    assert numpy_evpi(np.asarray(matrix, dtype=np.float64)) == pytest.approx(
        float(expected),
        rel=_tolerance(case, "numpy_relative"),
        abs=_tolerance(case, "numpy_absolute"),
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
        float(case["production_expected_evpi"]),
        rel=_tolerance(case, "production_relative"),
        abs=_tolerance(case, "production_absolute"),
    )
    assert actual == pytest.approx(
        float(case["expected_evpi"]),
        rel=_tolerance(case, "production_relative"),
        abs=_tolerance(case, "production_absolute"),
    )


def test_higher_dimensional_oracle_matches_polars_backend() -> None:
    pl = pytest.importorskip("polars")
    case = next(item for item in _cases() if item["case_id"] == "three-strategy")
    frame = pl.DataFrame(case["net_benefit"], orient="row")
    per_draw = frame.max_horizontal().mean()
    current = max(frame[column].mean() for column in frame.columns)
    assert per_draw - current == pytest.approx(
        float(case["expected_evpi"]),
        rel=_tolerance(case, "production_relative"),
        abs=_tolerance(case, "production_absolute"),
    )


def test_higher_dimensional_oracle_matches_optional_jax_backend() -> None:
    jnp = pytest.importorskip("jax.numpy")
    case = next(item for item in _cases() if item["case_id"] == "three-strategy")
    values = jnp.asarray(case["net_benefit"], dtype=jnp.float32)
    actual = jnp.mean(jnp.max(values, axis=1)) - jnp.max(jnp.mean(values, axis=0))
    assert float(actual) == pytest.approx(
        float(case["expected_evpi"]),
        abs=_tolerance(case, "jax_absolute"),
    )
