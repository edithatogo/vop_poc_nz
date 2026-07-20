"""Shared Pydantic configuration for immutable domain values."""

from pydantic import BaseModel, ConfigDict


class FrozenDomainModel(BaseModel):
    """Strict, immutable base for nested domain specifications.

    Deep immutability is achieved by requiring nested models and tuples; a
    frozen Pydantic model alone would not freeze contained lists or mappings.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )
