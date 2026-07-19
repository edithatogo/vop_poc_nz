"""Small pure predicates shared by fail-closed assurance gates."""

from __future__ import annotations

import math
from collections.abc import Collection


def has_exact_keys(actual: Collection[str], expected: Collection[str]) -> bool:
    """Return true only when no required or unknown semantic keys exist."""
    return set(actual) == set(expected)


def has_name_collision(names: Collection[str], existing: Collection[str] = ()) -> bool:
    """Return true for duplicates within names or collisions with existing names."""
    values = tuple(names)
    return len(set(values)) != len(values) or not set(values).isdisjoint(existing)


def exceeds_upper_bound(actual: float, maximum: float) -> bool:
    """Fail closed for invalid measurements or bounds and true regressions."""
    return (
        not math.isfinite(actual)
        or actual < 0.0
        or not math.isfinite(maximum)
        or maximum <= 0.0
        or actual > maximum
    )


__all__ = ["exceeds_upper_bound", "has_exact_keys", "has_name_collision"]
