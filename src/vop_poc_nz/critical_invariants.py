"""Pure C13 safety invariants suitable for strict mutation enforcement."""

from __future__ import annotations

from hmac import compare_digest
from typing import Literal

SocietalMethod = Literal["human_capital", "friction_cost"]


def require_matching_sha256(*, declared: str, actual: str, field: str) -> None:
    """Reject a forged or stale SHA-256 identity with a stable error."""
    if len(declared) != 64 or len(actual) != 64:
        raise ValueError(f"{field} must contain two SHA-256 digests")
    if not compare_digest(declared, actual):
        raise ValueError(f"{field} does not match calculated content")


def supported_societal_methods(
    *, has_human_capital: bool, has_friction_cost: bool
) -> tuple[SocietalMethod, ...]:
    """Return only societal methods supported by the validated input contract."""
    methods: list[SocietalMethod] = []
    if has_human_capital:
        methods.append("human_capital")
    if has_friction_cost:
        methods.append("friction_cost")
    return tuple(methods)


def require_societal_method_inputs(
    *,
    perspective: str,
    method: SocietalMethod,
    has_human_capital: bool,
    has_friction_cost: bool,
) -> None:
    """Fail closed when a requested societal method lacks required inputs."""
    if perspective != "societal":
        return
    if method == "human_capital" and not has_human_capital:
        raise ValueError("human-capital societal analysis requires productivity_costs")
    if method == "friction_cost" and not has_friction_cost:
        raise ValueError(
            "friction-cost societal analysis requires friction_cost_params"
        )


def exact_positive_int_or_none(value: object, *, field: str) -> int | None:
    """Accept an exact positive JSON integer or null without bool coercion."""
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer or null")
    return value


__all__ = [
    "SocietalMethod",
    "exact_positive_int_or_none",
    "require_matching_sha256",
    "require_societal_method_inputs",
    "supported_societal_methods",
]
