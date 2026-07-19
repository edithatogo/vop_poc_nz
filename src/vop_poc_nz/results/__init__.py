"""Immutable typed analysis results."""

from .base import AnalysisResult
from .cea import CEAAnalysisResult, ICERResult, ICERStatus

__all__ = ["AnalysisResult", "CEAAnalysisResult", "ICERResult", "ICERStatus"]
