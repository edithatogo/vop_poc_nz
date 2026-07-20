"""Minimal OTLP/HTTP JSON privacy and correlation contract."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_CAMEL_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_WORD_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_KEY_SEPARATOR = re.compile(r"[^a-z0-9]+")
_SENSITIVE_KEY_TOKENS = frozenset(
    {
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "jwt",
        "passphrase",
        "password",
        "secret",
        "session",
        "signature",
        "token",
    }
)
_SENSITIVE_COMPACT_KEYS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "clientsecret",
        "idtoken",
        "privatekey",
        "refreshtoken",
    }
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?:authorization|api[\s._-]*key|access[\s._-]*(?:key|token)|"
    r"client[\s._-]*secret|cookie|credentials?|jwt|passphrase|password|"
    r"private[\s._-]*key|refresh[\s._-]*token|secret|session|signature|token)"
    r"[\"']?\s*[:=]\s*[\"']?\S+",
    re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"\bbearer\s+\S+", re.IGNORECASE)
_REDACTED = "[REDACTED]"


class TelemetryContractError(ValueError):
    """Raised for unsafe or malformed telemetry exports."""


def _normalized_key_tokens(key: str) -> tuple[str, ...]:
    separated = _CAMEL_ACRONYM_BOUNDARY.sub(r"\1_\2", key)
    separated = _CAMEL_WORD_BOUNDARY.sub(r"\1_\2", separated).casefold()
    return tuple(part for part in _KEY_SEPARATOR.split(separated) if part)


def _is_sensitive_key(key: str) -> bool:
    tokens = _normalized_key_tokens(key)
    return (
        bool(_SENSITIVE_KEY_TOKENS.intersection(tokens))
        or "".join(tokens) in _SENSITIVE_COMPACT_KEYS
    )


def _contains_sensitive_pattern(text: str) -> bool:
    return bool(_SENSITIVE_ASSIGNMENT.search(text) or _BEARER_VALUE.search(text))


def _contains_sensitive_json(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _is_sensitive_key(str(key)) or _contains_sensitive_json(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_sensitive_json(item) for item in value)
    return isinstance(value, str) and _contains_sensitive_pattern(value)


def _contains_sensitive_text(text: str) -> bool:
    if _contains_sensitive_pattern(text):
        return True
    try:
        structured = json.loads(text)
    except json.JSONDecodeError, TypeError:
        return False
    return _contains_sensitive_json(structured)


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
        if any(_contains_sensitive_text(value) for value in text_fields):
            raise TelemetryContractError(
                "correlation fields contain secret-bearing text"
            )
        if _HEX_32.fullmatch(self.trace_id) is None:
            raise TelemetryContractError("trace_id must be 32 lowercase hex characters")
        if int(self.trace_id, 16) == 0:
            raise TelemetryContractError("trace_id must contain a non-zero byte")
        if _HEX_16.fullmatch(self.span_id) is None:
            raise TelemetryContractError("span_id must be 16 lowercase hex characters")
        if int(self.span_id, 16) == 0:
            raise TelemetryContractError("span_id must contain a non-zero byte")


def _safe_value(key: str, value: object) -> object:
    if _is_sensitive_key(key):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(name): _safe_value(str(name), item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_value(key, item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise TelemetryContractError("telemetry numbers must be finite")
    if isinstance(value, str) and _contains_sensitive_text(value):
        return _REDACTED
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    coerced = str(value)
    return _REDACTED if _contains_sensitive_text(coerced) else coerced


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
    if not isinstance(message, str) or not message.strip():
        raise TelemetryContractError("telemetry message must not be empty")
    if _contains_sensitive_text(message):
        raise TelemetryContractError("telemetry message contains secret-bearing text")
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
    timestamp = (
        time.time_ns() if observed_time_unix_nano is None else observed_time_unix_nano
    )
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or not 0 < timestamp < 2**64
    ):
        raise TelemetryContractError("observed timestamp must be a positive uint64")
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
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise TelemetryContractError("OTLP payload must be finite JSON") from exc
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
