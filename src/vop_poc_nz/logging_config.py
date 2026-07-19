"""Application-owned, context-aware logging configuration."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "vop_log_context", default=None
)
_OWNED_HANDLER = "_vop_poc_nz_handler"
_SENSITIVE_FRAGMENTS = ("authorization", "password", "secret", "token", "api_key")


def _safe_context(values: Mapping[str, object]) -> dict[str, str]:
    return {
        key: "[REDACTED]"
        if any(fragment in key.casefold() for fragment in _SENSITIVE_FRAGMENTS)
        else str(value)
        for key, value in values.items()
    }


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
        record.service = self.settings.service
        record.run_id = self.settings.run_id
        record.context = dict(_CONTEXT.get() or {})
        return True


class JsonFormatter(logging.Formatter):
    """Emit newline-delimited JSON suitable for CI artifact ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", "vop_poc_nz"),
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "context", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


@contextmanager
def log_context(**values: object) -> Iterator[None]:
    """Bind serializable context to logs within the current async/task context."""
    merged = {
        **(_CONTEXT.get() or {}),
        **_safe_context(values),
    }
    token = _CONTEXT.set(merged)
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
    human = logging.Formatter(
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
