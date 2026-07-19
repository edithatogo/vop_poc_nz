"""Application-owned, context-aware logging configuration."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

_OWNED_HANDLER = "_vop_poc_nz_handler"
_SENSITIVE_FRAGMENTS = ("authorization", "password", "secret", "token", "api_key")
_RESERVED_FIELDS = frozenset(
    {
        "analysis_id",
        "backend_requested",
        "backend_selected",
        "exception",
        "fallback_code",
        "level",
        "logger",
        "message",
        "numerical_policy_id",
        "run_id",
        "service",
        "span_id",
        "timestamp",
        "trace_flags",
        "trace_id",
        "traceparent",
    }
)
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_TRACE_FLAGS_RE = re.compile(r"^[0-9a-f]{2}$")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|password|secret|token|api[_-]?key)\s*([:=])\s*([^\s,;]+)"
)


@dataclass(frozen=True)
class _LogContextState:
    run_id: str | None
    correlation: Mapping[str, object]
    fields: Mapping[str, object]


_CONTEXT: ContextVar[_LogContextState | None] = ContextVar(
    "vop_log_context", default=None
)


def _current_context() -> _LogContextState:
    return _CONTEXT.get() or _LogContextState(run_id=None, correlation={}, fields={})


def _redact_text(value: str) -> str:
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    return _ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", value)


def _sensitive_key(key: str) -> bool:
    folded = key.casefold()
    return any(fragment in folded for fragment in _SENSITIVE_FRAGMENTS)


def _redact_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _sensitive_key(str(key)) else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


def _safe_context(values: Mapping[str, object]) -> dict[str, object]:
    reserved = sorted(_RESERVED_FIELDS.intersection(values))
    if reserved:
        raise ValueError(f"reserved logging context field: {reserved[0]}")
    return {
        key: "[REDACTED]" if _sensitive_key(key) else _redact_value(value)
        for key, value in values.items()
    }


class TraceContext(BaseModel):
    """W3C Trace Context identifiers for one correlated execution span."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(default_factory=lambda: secrets.token_hex(16))
    span_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    trace_flags: str = "00"

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        if _TRACE_ID_RE.fullmatch(value) is None or value == "0" * 32:
            raise ValueError("trace_id must be 32 lowercase non-zero hex characters")
        return value

    @field_validator("span_id")
    @classmethod
    def validate_span_id(cls, value: str) -> str:
        if _SPAN_ID_RE.fullmatch(value) is None or value == "0" * 16:
            raise ValueError("span_id must be 16 lowercase non-zero hex characters")
        return value

    @field_validator("trace_flags")
    @classmethod
    def validate_trace_flags(cls, value: str) -> str:
        if _TRACE_FLAGS_RE.fullmatch(value) is None:
            raise ValueError("trace_flags must be two lowercase hex characters")
        return value

    @property
    def traceparent(self) -> str:
        """Return the canonical W3C ``traceparent`` header value."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"


class AnalysisLogContext(BaseModel):
    """Trusted correlation fields bound at a typed analysis boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    trace: TraceContext = Field(default_factory=TraceContext)
    analysis_id: str = Field(min_length=1)
    backend_requested: str = Field(min_length=1)
    backend_selected: str = Field(min_length=1)
    fallback_code: str = Field(min_length=1)
    numerical_policy_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    def correlation_fields(self) -> dict[str, str]:
        """Project immutable identifiers to the canonical log-event shape."""
        return {
            "analysis_id": self.analysis_id,
            "backend_requested": self.backend_requested,
            "backend_selected": self.backend_selected,
            "fallback_code": self.fallback_code,
            "numerical_policy_id": self.numerical_policy_id,
            "trace_id": self.trace.trace_id,
            "span_id": self.trace.span_id,
            "trace_flags": self.trace.trace_flags,
            "traceparent": self.trace.traceparent,
        }


def numerical_policy_digest(policy: BaseModel | Mapping[str, object]) -> str:
    """Return a deterministic identifier without logging policy values."""
    value: object = (
        policy.model_dump(mode="json") if isinstance(policy, BaseModel) else policy
    )
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class LoggingSettings(BaseModel):
    """Validated settings for deterministic CLI and library logging."""

    model_config = ConfigDict(extra="forbid", frozen=True, defer_build=True)

    level: str = "INFO"
    json_output: bool = False
    console: bool = True
    log_file: Path | None = None
    service: str = "vop_poc_nz"
    run_id: str = Field(default_factory=lambda: uuid4().hex)

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown logging level: {value}")
        return normalized

    @classmethod
    def from_environment(cls, **overrides: Any) -> LoggingSettings:
        """Load the stable ``VOP_LOG_*`` environment contract."""
        values: dict[str, Any] = {
            "level": os.getenv("VOP_LOG_LEVEL", "INFO"),
            "json_output": os.getenv("VOP_LOG_FORMAT", "human").lower() == "json",
            "run_id": os.getenv("VOP_RUN_ID") or uuid4().hex,
        }
        values.update(overrides)
        return cls.model_validate(values)


class _ContextFilter(logging.Filter):
    def __init__(self, settings: LoggingSettings) -> None:
        super().__init__()
        self.settings = settings

    def filter(self, record: logging.LogRecord) -> bool:
        state = _current_context()
        record.service = self.settings.service
        record.run_id = state.run_id or self.settings.run_id
        record.correlation = dict(state.correlation)
        record.context = dict(state.fields)
        return True


class RedactingFormatter(logging.Formatter):
    """Apply the same credential redaction policy to human-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        return _redact_text(super().format(record))


class JsonFormatter(logging.Formatter):
    """Emit newline-delimited JSON suitable for CI artifact ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", "vop_poc_nz"),
            "run_id": getattr(record, "run_id", None),
            "message": _redact_text(record.getMessage()),
        }
        payload.update(getattr(record, "correlation", {}))
        payload.update(getattr(record, "context", {}))
        if record.exc_info:
            payload["exception"] = _redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


@contextmanager
def log_context(**values: object) -> Iterator[None]:
    """Bind serializable context to logs within the current async/task context."""
    current = _current_context()
    state = _LogContextState(
        run_id=current.run_id,
        correlation=current.correlation,
        fields={**current.fields, **_safe_context(values)},
    )
    token = _CONTEXT.set(state)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


@contextmanager
def analysis_log_context(context: AnalysisLogContext) -> Iterator[None]:
    """Bind trusted run/trace/backend/policy correlation for one analysis."""
    current = _current_context()
    state = _LogContextState(
        run_id=context.run_id,
        correlation=context.correlation_fields(),
        fields=current.fields,
    )
    token = _CONTEXT.set(state)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def configure_logging(settings: LoggingSettings | None = None) -> logging.Logger:
    """Configure only the package logger and return it.

    Reconfiguration removes handlers created by this function while preserving
    application and third-party handlers on the root logger.
    """
    settings = settings or LoggingSettings.from_environment()
    logger = logging.getLogger("vop_poc_nz")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        if getattr(handler, _OWNED_HANDLER, False):
            root = logging.getLogger()
            if handler in root.handlers:
                root.removeHandler(handler)
            logger.removeHandler(handler)
            handler.close()

    context_filter = _ContextFilter(settings)
    human = RedactingFormatter(
        "%(asctime)s %(levelname)s %(name)s [run_id=%(run_id)s] %(message)s"
    )
    formatter: logging.Formatter = JsonFormatter() if settings.json_output else human

    if settings.console:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(settings.level)
        console.setFormatter(formatter)
        console.addFilter(context_filter)
        setattr(console, _OWNED_HANDLER, True)
        logger.addHandler(console)

    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            settings.log_file, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(context_filter)
        setattr(file_handler, _OWNED_HANDLER, True)
        logger.addHandler(file_handler)

    logger.info(
        "logging_configured", extra={"configuration": settings.model_dump(mode="json")}
    )
    return logger


def setup_logging(
    output_dir: str | os.PathLike[str] | None = None,
    log_file: str = "analysis.log",
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Backward-compatible adapter for the historical analysis entrypoints."""
    resolved_level = logging.getLevelName(level) if isinstance(level, int) else level
    path = Path(output_dir or ".") / log_file
    package_logger = configure_logging(
        LoggingSettings.from_environment(level=resolved_level, log_file=path)
    )
    root = logging.getLogger()
    for handler in tuple(root.handlers):
        if getattr(handler, _OWNED_HANDLER, False):
            root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    for handler in package_logger.handlers:
        root.addHandler(handler)
    return package_logger


def logging_metadata(settings: LoggingSettings) -> Mapping[str, Any]:
    """Return safe, JSON-ready run metadata for manifests."""
    return settings.model_dump(mode="json")
