"""Minimal OTLP/HTTP JSON privacy and correlation contract."""

from __future__ import annotations

import ipaddress
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_SENSITIVE_KEY = re.compile(
    r"(^|[._-])(authorization|api[_-]?key|cookie|password|secret|token)($|[._-])",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


class TelemetryContractError(ValueError):
    """Raised for unsafe or malformed telemetry exports."""


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Required analysis correlation carried by every C15 OTLP record."""

    run_id: str
    trace_id: str
    span_id: str
    backend: str
    fallback: str
    numerical_policy_id: str

    def __post_init__(self) -> None:
        text_fields = (
            self.run_id,
            self.backend,
            self.fallback,
            self.numerical_policy_id,
        )
        if any(not value.strip() for value in text_fields):
            raise TelemetryContractError("correlation fields must not be empty")
        if _HEX_32.fullmatch(self.trace_id) is None:
            raise TelemetryContractError("trace_id must be 32 lowercase hex characters")
        if _HEX_16.fullmatch(self.span_id) is None:
            raise TelemetryContractError("span_id must be 16 lowercase hex characters")


def _safe_value(key: str, value: object) -> object:
    if _SENSITIVE_KEY.search(key):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(name): _safe_value(str(name), item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_value(key, item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _any_value(value: object) -> dict[str, object]:
    if value is None:
        return {"stringValue": "null"}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    if isinstance(value, Mapping):
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(key), "value": _any_value(item)}
                    for key, item in sorted(value.items())
                ]
            }
        }
    raise TelemetryContractError("unsupported telemetry value")


def build_otlp_log_request(
    message: str,
    correlation: CorrelationContext,
    *,
    attributes: Mapping[str, object] | None = None,
    observed_time_unix_nano: int | None = None,
) -> dict[str, object]:
    """Build an OTLP/HTTP JSON logs request with mandatory safe correlation."""
    if not message.strip():
        raise TelemetryContractError("telemetry message must not be empty")
    safe = {
        str(key): _safe_value(str(key), value)
        for key, value in (attributes or {}).items()
    }
    safe.update(
        {
            "run.id": correlation.run_id,
            "analysis.backend": correlation.backend,
            "analysis.fallback": correlation.fallback,
            "analysis.numerical_policy_id": correlation.numerical_policy_id,
        }
    )
    timestamp = observed_time_unix_nano or time.time_ns()
    record = {
        "timeUnixNano": str(timestamp),
        "observedTimeUnixNano": str(timestamp),
        "severityText": "INFO",
        "body": {"stringValue": message},
        "traceId": correlation.trace_id,
        "spanId": correlation.span_id,
        "attributes": [
            {"key": key, "value": _any_value(value)}
            for key, value in sorted(safe.items())
        ],
    }
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "vop_poc_nz"},
                        }
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "vop_poc_nz.c15", "version": "1.0.0"},
                        "logRecords": [record],
                    }
                ],
            }
        ]
    }


def _safe_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.path != "/v1/logs" or parsed.query or parsed.fragment:
        raise TelemetryContractError("OTLP endpoint must use the exact /v1/logs path")
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname:
        try:
            if ipaddress.ip_address(parsed.hostname).is_loopback:
                return
        except ValueError:
            if parsed.hostname.casefold() == "localhost":
                return
    raise TelemetryContractError("OTLP export requires HTTPS or loopback HTTP")


def export_otlp_http(
    endpoint: str, payload: Mapping[str, object], *, timeout: float = 5.0
) -> None:
    """POST one privacy-screened OTLP JSON request to a collector endpoint."""
    _safe_endpoint(endpoint)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    request = Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # nosec B310
        if not 200 <= response.status < 300:
            raise TelemetryContractError(
                f"collector returned unexpected HTTP status {response.status}"
            )


__all__ = [
    "CorrelationContext",
    "TelemetryContractError",
    "build_otlp_log_request",
    "export_otlp_http",
]
