"""Immutable typed analysis results."""

from .base import (
    AnalysisResult,
    AnalysisResultEnvelope,
    ArrowSchemaIdentity,
    DiagnosticSeverity,
    ResultDiagnostic,
    ResultMaturity,
    ResultMetadata,
)
from .cea import CEAAnalysisResult, ICERResult, ICERStatus

__all__ = [
    "AnalysisResult",
    "AnalysisResultEnvelope",
    "ArrowSchemaIdentity",
    "CEAAnalysisResult",
    "DiagnosticSeverity",
    "ICERResult",
    "ICERStatus",
    "ResultDiagnostic",
    "ResultMaturity",
    "ResultMetadata",
]
