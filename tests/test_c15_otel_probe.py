from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import c15_otel_probe
from scripts.c15_otel_probe import _received_contract, probe
from vop_poc_nz.c15_otel_contract import CorrelationContext, build_otlp_log_request


def _context() -> CorrelationContext:
    return CorrelationContext(
        run_id="run",
        trace_id="0123456789abcdef0123456789abcdef",
        span_id="0123456789abcdef",
        backend="numpy",
        fallback="none",
        numerical_policy_id="policy",
    )


def test_probe_reports_collector_observed_correlation() -> None:
    report = probe()
    assert report["correlation_source"] == "collector_received_payload"
    assert report["privacy_screened"] is True


def test_collector_parser_rejects_received_correlation_drift() -> None:
    context = _context()
    payload = build_otlp_log_request(
        "analysis.completed", context, attributes={"safe": "retained"}
    )
    drifted = deepcopy(payload)
    drifted["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["traceId"] = "f" * 32
    with pytest.raises(RuntimeError, match="correlation"):
        _received_contract(drifted, context)


def test_probe_cli_writes_structured_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "otel-failure.json"

    def fail_probe() -> dict[str, object]:
        raise RuntimeError("collector unavailable")

    monkeypatch.setattr(c15_otel_probe, "probe", fail_probe)
    monkeypatch.setattr(sys, "argv", ["c15_otel_probe.py", "--output", str(output)])
    assert c15_otel_probe.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["privacy_screened"] is False
