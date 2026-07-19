"""Exhaustive tests for the mutation-gated C13 safety decisions."""

from __future__ import annotations

import pytest

from vop_poc_nz.critical_invariants import (
    SocietalMethod,
    exact_positive_int_or_none,
    require_matching_sha256,
    require_societal_method_inputs,
    supported_societal_methods,
)

ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64


def test_matching_sha256_accepts_exact_identity() -> None:
    assert (
        require_matching_sha256(
            declared=ZERO_SHA256, actual=ZERO_SHA256, field="artifact"
        )
        is None
    )


@pytest.mark.parametrize(
    ("declared", "actual"),
    [("0" * 63, ZERO_SHA256), (ZERO_SHA256, "0" * 63)],
)
def test_matching_sha256_rejects_invalid_lengths(declared: str, actual: str) -> None:
    with pytest.raises(
        ValueError, match=r"^artifact must contain two SHA-256 digests$"
    ):
        require_matching_sha256(declared=declared, actual=actual, field="artifact")


def test_matching_sha256_rejects_different_digest() -> None:
    with pytest.raises(
        ValueError, match=r"^artifact does not match calculated content$"
    ):
        require_matching_sha256(
            declared=ZERO_SHA256, actual=ONE_SHA256, field="artifact"
        )


@pytest.mark.parametrize(
    ("human", "friction", "expected"),
    [
        (False, False, ()),
        (True, False, ("human_capital",)),
        (False, True, ("friction_cost",)),
        (True, True, ("human_capital", "friction_cost")),
    ],
)
def test_supported_societal_methods_are_capability_derived(
    human: bool, friction: bool, expected: tuple[str, ...]
) -> None:
    assert (
        supported_societal_methods(has_human_capital=human, has_friction_cost=friction)
        == expected
    )


@pytest.mark.parametrize("method", ["human_capital", "friction_cost"])
def test_non_societal_analysis_does_not_require_productivity_inputs(
    method: SocietalMethod,
) -> None:
    require_societal_method_inputs(
        perspective="health_system",
        method=method,
        has_human_capital=False,
        has_friction_cost=False,
    )


@pytest.mark.parametrize(
    ("method", "human", "friction"),
    [("human_capital", True, False), ("friction_cost", False, True)],
)
def test_societal_analysis_accepts_available_method(
    method: SocietalMethod, human: bool, friction: bool
) -> None:
    require_societal_method_inputs(
        perspective="societal",
        method=method,
        has_human_capital=human,
        has_friction_cost=friction,
    )


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (
            "human_capital",
            "human-capital societal analysis requires productivity_costs",
        ),
        (
            "friction_cost",
            "friction-cost societal analysis requires friction_cost_params",
        ),
    ],
)
def test_societal_analysis_rejects_missing_method_input(
    method: SocietalMethod, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        require_societal_method_inputs(
            perspective="societal",
            method=method,
            has_human_capital=False,
            has_friction_cost=False,
        )


@pytest.mark.parametrize("value", [1, 28, 2**31 - 1])
def test_exact_positive_integer_accepts_positive_int(value: int) -> None:
    assert exact_positive_int_or_none(value, field="issue_number") == value


def test_exact_positive_integer_accepts_none() -> None:
    assert exact_positive_int_or_none(None, field="issue_number") is None


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", [], {}])
def test_exact_positive_integer_rejects_coercive_values(value: object) -> None:
    with pytest.raises(
        ValueError, match=r"^issue_number must be a positive integer or null$"
    ):
        exact_positive_int_or_none(value, field="issue_number")
