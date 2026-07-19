from __future__ import annotations

import json
import logging
from importlib.metadata import version

import pytest
from pydantic import ValidationError

import vop_poc_nz
from vop_poc_nz.logging_config import LoggingSettings, configure_logging, log_context


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
