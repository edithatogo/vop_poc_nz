from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import pytest
from pydantic import ValidationError

import vop_poc_nz
from vop_poc_nz.logging_config import (
    AnalysisLogContext,
    LoggingSettings,
    TraceContext,
    analysis_log_context,
    configure_logging,
    log_context,
    numerical_policy_digest,
)


def test_runtime_version_matches_distribution_metadata() -> None:
    assert vop_poc_nz.__version__ == version("vop_poc_nz")


def test_logging_settings_reject_unknown_level() -> None:
    with pytest.raises(ValidationError):
        LoggingSettings(level="verbose")


def test_json_logging_carries_run_and_bound_context(tmp_path) -> None:
    destination = tmp_path / "run.jsonl"
    logger = configure_logging(
        LoggingSettings(
            console=False,
            json_output=True,
            log_file=destination,
            run_id="test-run",
        )
    )
    with log_context(track="C11", dataset="fixture", api_key="never-log-this"):
        logger.info("profile_started")
    for handler in logger.handlers:
        handler.flush()

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert records[-1]["message"] == "profile_started"
    assert records[-1]["run_id"] == "test-run"
    assert records[-1]["track"] == "C11"
    assert records[-1]["api_key"] == "[REDACTED]"


def test_configuration_preserves_root_handlers() -> None:
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    try:
        configure_logging(LoggingSettings(console=False))
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_trace_context_is_w3c_compatible_and_rejects_invalid_identifiers() -> None:
    trace = TraceContext(
        trace_id="1" * 32,
        span_id="2" * 16,
        trace_flags="01",
    )
    assert trace.traceparent == f"00-{'1' * 32}-{'2' * 16}-01"

    for field, value in (
        ("trace_id", "0" * 32),
        ("trace_id", "A" * 32),
        ("span_id", "0" * 16),
        ("span_id", "2" * 15),
        ("trace_flags", "zz"),
    ):
        with pytest.raises(ValidationError):
            TraceContext.model_validate(
                {"trace_id": "1" * 32, "span_id": "2" * 16, field: value}
            )


def test_analysis_context_correlates_and_nested_context_restores(tmp_path) -> None:
    destination = tmp_path / "correlated.jsonl"
    logger = configure_logging(
        LoggingSettings(console=False, log_file=destination, run_id="settings-run")
    )
    policy_id = numerical_policy_digest({"dtype": "float64", "rtol": 1e-9})
    context = AnalysisLogContext(
        run_id="analysis-run",
        trace=TraceContext(trace_id="1" * 32, span_id="2" * 16),
        analysis_id="analysis-1",
        backend_requested="numpy",
        backend_selected="numpy",
        fallback_code="none",
        numerical_policy_id=policy_id,
    )

    with analysis_log_context(context):
        logger.info("outer")
        with log_context(stage="calculation"):
            logger.info("inner")
        logger.info("restored")
    for handler in logger.handlers:
        handler.flush()

    outer, inner, restored = _records(destination)[-3:]
    for record in (outer, inner, restored):
        assert record["run_id"] == "analysis-run"
        assert record["trace_id"] == "1" * 32
        assert record["span_id"] == "2" * 16
        assert record["traceparent"] == f"00-{'1' * 32}-{'2' * 16}-00"
        assert record["backend_selected"] == "numpy"
        assert record["numerical_policy_id"] == policy_id
    assert "stage" not in outer
    assert inner["stage"] == "calculation"
    assert "stage" not in restored


def test_context_redaction_is_recursive_and_reserved_fields_fail_closed(
    tmp_path,
) -> None:
    destination = tmp_path / "redacted.jsonl"
    logger = configure_logging(
        LoggingSettings(console=False, log_file=destination, run_id="safe-run")
    )

    for reserved in ("message", "run_id", "trace_id", "backend_selected"):
        with (
            pytest.raises(ValueError, match="reserved logging context field"),
            log_context(**{reserved: "forged"}),
        ):
            pass

    with log_context(
        request={
            "headers": {"Authorization": "Bearer top-secret"},
            "items": [{"api_token": "nested-secret"}],
        },
        note="password=hunter2",
    ):
        logger.error("request failed with Bearer message-secret")
    for handler in logger.handlers:
        handler.flush()

    rendered = destination.read_text(encoding="utf-8")
    assert "top-secret" not in rendered
    assert "nested-secret" not in rendered
    assert "hunter2" not in rendered
    assert "message-secret" not in rendered
    record = _records(destination)[-1]
    assert record["request"] == {
        "headers": {"Authorization": "[REDACTED]"},
        "items": [{"api_token": "[REDACTED]"}],
    }
