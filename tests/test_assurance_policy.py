from __future__ import annotations

import math

import pytest

from vop_poc_nz.assurance_policy import (
    exceeds_upper_bound,
    has_exact_keys,
    has_name_collision,
)


def test_exact_keys_rejects_missing_and_unknown_semantics() -> None:
    assert has_exact_keys(("a", "b"), ("b", "a")) is True
    assert has_exact_keys(("a",), ("a", "b")) is False
    assert has_exact_keys(("a", "b"), ("a",)) is False


def test_name_collision_detects_internal_and_existing_names() -> None:
    assert has_name_collision(("a", "b"), ("old",)) is False
    assert has_name_collision(("a", "a")) is True
    assert has_name_collision(("a", "b"), ("b", "old")) is True


@pytest.mark.parametrize(
    ("actual", "maximum", "expected"),
    [
        (0.0, 1.0, False),
        (1.0, 1.0, False),
        (1.0001, 1.0, True),
        (-0.1, 1.0, True),
        (math.inf, 1.0, True),
        (math.nan, 1.0, True),
        (1.0, 0.0, True),
        (1.0, -1.0, True),
        (1.0, math.inf, True),
        (1.0, math.nan, True),
    ],
)
def test_upper_bound_is_inclusive_and_fails_closed(
    actual: float, maximum: float, expected: bool
) -> None:
    assert exceeds_upper_bound(actual, maximum) is expected
