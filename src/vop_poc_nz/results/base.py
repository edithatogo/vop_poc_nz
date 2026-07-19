"""Shared analysis-result contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AnalysisResult(Protocol):
    analysis_type: str
    contract_version: str

    def to_legacy_dict(self) -> dict[str, object]: ...
