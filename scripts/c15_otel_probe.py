#!/usr/bin/env python3
"""Exercise an OTLP/HTTP export against an ephemeral local collector simulator."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

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
    encoded = json.dumps(_Collector.payloads, sort_keys=True)
    if len(_Collector.payloads) != 1 or "must-not-export" in encoded:
        raise RuntimeError("ephemeral collector privacy contract failed")
    return {
        "schema_version": "1.0.0",
        "collector": "ephemeral-loopback-otlp-http-json-simulator",
        "exports_received": 1,
        "privacy_screened": True,
        "secret_body_rejected": secret_body_rejected,
        "correlation": {
            "run_id": context.run_id,
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "backend": context.backend,
            "fallback": context.fallback,
            "numerical_policy_id": context.numerical_policy_id,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = probe()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"C15 telemetry assurance failed: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
