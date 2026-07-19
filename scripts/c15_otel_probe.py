#!/usr/bin/env python3
"""Exercise an OTLP/HTTP export against an ephemeral local collector simulator."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast

from vop_poc_nz.c15_otel_contract import (
    CorrelationContext,
    TelemetryContractError,
    build_otlp_log_request,
    export_otlp_http,
)


class _Collector(BaseHTTPRequestHandler):
    payloads: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        if (
            self.path != "/v1/logs"
            or self.headers.get("Content-Type") != "application/json"
        ):
            self.send_error(400)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            self.send_error(400)
            return
        self.payloads.append(payload)
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *_args: object) -> None:
        del format
        return


def _received_contract(
    payload: dict[str, object], expected: CorrelationContext
) -> dict[str, object]:
    """Independently parse collector bytes and validate every required field."""
    try:
        resource_logs = cast("list[dict[str, object]]", payload["resourceLogs"])
        resource = resource_logs[0]
        scope_logs = cast("list[dict[str, object]]", resource["scopeLogs"])
        records = cast("list[dict[str, object]]", scope_logs[0]["logRecords"])
        record = records[0]
        raw_attributes = cast("list[dict[str, object]]", record["attributes"])
        attributes: dict[str, object] = {}
        for item in raw_attributes:
            key = cast("str", item["key"])
            value = cast("dict[str, object]", item["value"])
            attributes[key] = next(iter(value.values()))
    except (IndexError, KeyError, StopIteration, TypeError) as exc:
        raise RuntimeError(
            "collector received malformed OTLP correlation payload"
        ) from exc
    if len(resource_logs) != 1 or len(records) != 1:
        raise RuntimeError("collector must receive exactly one OTLP log record")
    observed = {
        "run_id": attributes.get("run.id"),
        "trace_id": record.get("traceId"),
        "span_id": record.get("spanId"),
        "backend": attributes.get("analysis.backend"),
        "fallback": attributes.get("analysis.fallback"),
        "numerical_policy_id": attributes.get("analysis.numerical_policy_id"),
    }
    required = {
        "run_id": expected.run_id,
        "trace_id": expected.trace_id,
        "span_id": expected.span_id,
        "backend": expected.backend,
        "fallback": expected.fallback,
        "numerical_policy_id": expected.numerical_policy_id,
    }
    if observed != required:
        raise RuntimeError("collector-observed correlation does not match the export")
    encoded = json.dumps(payload, sort_keys=True).casefold()
    if "must-not-export" in encoded:
        raise RuntimeError("collector received privacy-sensitive telemetry")
    if attributes.get("authorization") != "[REDACTED]":
        raise RuntimeError("collector did not receive the required redaction marker")
    if attributes.get("safe") != "retained":
        raise RuntimeError("collector did not receive the expected safe attribute")
    return observed


def probe() -> dict[str, object]:
    """Export one correlated record and return privacy-bounded probe evidence."""
    _Collector.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = CorrelationContext(
        run_id="c15-collector-probe",
        trace_id="0123456789abcdef0123456789abcdef",
        span_id="0123456789abcdef",
        backend="numpy",
        fallback="none",
        numerical_policy_id="c15-policy",
    )
    try:
        build_otlp_log_request(
            "Authorization: Bearer must-not-export-body",
            context,
        )
    except TelemetryContractError:
        secret_body_rejected = True
    else:
        raise RuntimeError("secret-bearing telemetry body was accepted")
    try:
        payload = build_otlp_log_request(
            "analysis.completed",
            context,
            attributes={"authorization": "Bearer must-not-export", "safe": "retained"},
        )
        export_otlp_http(f"http://127.0.0.1:{server.server_port}/v1/logs", payload)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    if len(_Collector.payloads) != 1:
        raise RuntimeError("ephemeral collector privacy contract failed")
    observed = _received_contract(_Collector.payloads[0], context)
    return {
        "schema_version": "1.0.0",
        "collector": "ephemeral-loopback-otlp-http-json-simulator",
        "exports_received": 1,
        "privacy_screened": True,
        "secret_body_rejected": secret_body_rejected,
        "correlation": observed,
        "correlation_source": "collector_received_payload",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = probe()
    except (OSError, RuntimeError, ValueError) as exc:
        report = {
            "schema_version": "1.0.0",
            "status": "failure",
            "privacy_screened": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(payload, encoding="utf-8", newline="\n")
        print(payload, end="")
        return 2
    report["status"] = "success"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
