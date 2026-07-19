from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vop_poc_nz.c15_otel_contract import (
    CorrelationContext,
    TelemetryContractError,
    build_otlp_log_request,
    export_otlp_http,
)


def _context() -> CorrelationContext:
    return CorrelationContext(
        run_id="run-c15",
        trace_id="0123456789abcdef0123456789abcdef",
        span_id="0123456789abcdef",
        backend="numpy",
        fallback="none",
        numerical_policy_id="policy-sha256",
    )


def test_otlp_request_preserves_correlation_and_redacts_nested_secrets() -> None:
    payload = build_otlp_log_request(
        "analysis.completed",
        _context(),
        attributes={
            "cohort": "public",
            "authorization": "Bearer secret",
            "nested": {"api_key": "private", "safe": ["ok", {"token": "no"}]},
        },
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert "Bearer secret" not in encoded
    assert "private" not in encoded
    assert '"stringValue": "[REDACTED]"' in encoded
    record = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert record["traceId"] == _context().trace_id
    assert record["spanId"] == _context().span_id
    attribute_keys = {item["key"] for item in record["attributes"]}
    assert {
        "run.id",
        "analysis.backend",
        "analysis.fallback",
        "analysis.numerical_policy_id",
    } <= attribute_keys


def test_ephemeral_collector_simulator_receives_otlp_json() -> None:
    received: list[tuple[str, str, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            received.append(
                (self.path, self.headers["Content-Type"], self.rfile.read(length))
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *_args: object) -> None:
            del format
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        export_otlp_http(
            f"http://127.0.0.1:{server.server_port}/v1/logs",
            build_otlp_log_request("analysis.completed", _context()),
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert received[0][:2] == ("/v1/logs", "application/json")
    assert json.loads(received[0][2])["resourceLogs"]


def test_export_rejects_non_loopback_plain_http() -> None:
    with pytest.raises(TelemetryContractError, match="HTTPS or loopback"):
        export_otlp_http(
            "http://example.com/v1/logs",
            build_otlp_log_request("analysis.completed", _context()),
        )
