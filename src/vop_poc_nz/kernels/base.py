"""Generic contracts shared by pure calculation boundaries."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from vop_poc_nz.domain.base import FrozenDomainModel


class CalculationContext(FrozenDomainModel):
    case_id: str | None = None
    seed: int | None = None


SpecT = TypeVar("SpecT", contravariant=True)
ContextT = TypeVar("ContextT", bound=CalculationContext, contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


@runtime_checkable
class CalculationKernel(Protocol[SpecT, ContextT, ResultT]):
    """Structural contract for deterministic, typed analysis kernels."""

    name: str
    contract_version: str

    def calculate(self, spec: SpecT, *, context: ContextT) -> ResultT: ...
