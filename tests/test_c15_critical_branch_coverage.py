from __future__ import annotations

import json
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from tests.test_c15_performance import (
    _CONFIG_DIGEST,
    _baseline,
    _measurement,
    _runner,
)
from vop_poc_nz.c15_otel_contract import (
    CorrelationContext,
    TelemetryContractError,
    _any_value,
    _contains_sensitive_json,
    _safe_endpoint,
    build_otlp_log_request,
)
from vop_poc_nz.c15_performance import (
    _percentile,
    confidence_interval,
    normalized_runner_identity,
    performance_budget,
    performance_config_digest,
    performance_ratchet,
    runner_fingerprint,
)
from vop_poc_nz.c15_reproducibility import (
    ArtifactMismatch,
    compare_digest_reports,
    normalized_archive_report,
)
from vop_poc_nz.c15_scientific_oracles import decimal_evpi, numpy_evpi


@pytest.mark.parametrize(
    "value",
    [None, "text", [1], [[1], [1, 2]], [[1, "NaN"]]],
)
def test_decimal_oracle_rejects_every_invalid_matrix_surface(value: object) -> None:
    with pytest.raises(ValueError):
        decimal_evpi(value)


@pytest.mark.parametrize("value", [[], [[1]], [[1, 2], [3, float("inf")]]])
def test_numpy_oracle_rejects_invalid_or_nonfinite_matrix(value: object) -> None:
    with pytest.raises(ValueError):
        numpy_evpi(value)


@pytest.mark.parametrize(
    ("samples", "confidence", "resamples"),
    [
        ([1.0], 0.95, 1_000),
        ([1.0, 2.0], 1.0, 1_000),
        ([1.0, -1.0], 0.95, 1_000),
        ([1.0] * 5, 0.95, 999),
    ],
)
def test_confidence_interval_rejects_invalid_contract(
    samples: list[float], confidence: float, resamples: int
) -> None:
    with pytest.raises(ValueError):
        confidence_interval(
            samples, confidence=confidence, bootstrap_resamples=resamples
        )


def test_confidence_interval_normal_fallback_and_exact_percentile() -> None:
    interval = confidence_interval([1.0, 1.1, 0.9, 1.0])
    assert interval.method == "normal"
    assert interval.resamples == 0
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_performance_identity_and_configuration_validation_branches() -> None:
    with pytest.raises(ValueError, match="missing"):
        normalized_runner_identity({})
    with pytest.raises(ValueError, match="numeric"):
        normalized_runner_identity(_runner(python="latest"))
    with pytest.raises(ValueError, match="empty"):
        runner_fingerprint({})
    with pytest.raises(ValueError, match="workload"):
        performance_config_digest(workload_id="", parameters={}, confidence=0.95)


@pytest.mark.parametrize(
    ("maximum", "baseline", "digest", "message"),
    [
        (1.0, _baseline(), _CONFIG_DIGEST, "baseline or an absolute"),
        (None, _baseline(), None, "config digest"),
        (None, None, None, "baseline or absolute"),
        (0.0, None, None, "positive and finite"),
    ],
)
def test_performance_ratchet_rejects_ambiguous_or_invalid_budget(
    maximum: float | None,
    baseline: dict[str, object] | None,
    digest: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        performance_ratchet(
            [0.1] * 5,
            maximum_upper_seconds=maximum,
            baseline=baseline,
            runner=_runner(),
            config_digest=digest,
        )


def test_performance_baseline_structural_validation_branches() -> None:
    mutations: list[tuple[dict[str, object], str]] = []
    unsupported = _baseline()
    unsupported["schema_version"] = "0"
    mutations.append((unsupported, "schema"))
    no_cohorts = _baseline()
    no_cohorts["cohorts"] = []
    mutations.append((no_cohorts, "cohorts"))
    no_identity = _baseline()
    cast(dict[str, object], no_identity["cohorts"])["Linux"] = {}
    mutations.append((no_identity, "approved runner identity"))
    invalid_measurement = _baseline()
    invalid_measurement["measurement"] = []
    mutations.append((invalid_measurement, "measurement configuration"))
    invalid_source = _baseline()
    invalid_source["source"] = {"commit": ""}
    mutations.append((invalid_source, "source evidence"))
    for baseline, message in mutations:
        with pytest.raises(ValueError, match=message):
            performance_budget(baseline, runner=_runner(), measurement=_measurement())


@pytest.mark.parametrize(
    "field",
    [
        "reference_upper_seconds",
        "maximum_regression_factor",
        "absolute_ceiling_seconds",
    ],
)
@pytest.mark.parametrize("value", [True, 0.0])
def test_performance_baseline_rejects_invalid_positive_values(
    field: str, value: object
) -> None:
    baseline = _baseline()
    cohort = cast(
        dict[str, object], cast(dict[str, object], baseline["cohorts"])["Linux"]
    )
    cohort[field] = value
    with pytest.raises(ValueError):
        performance_budget(baseline, runner=_runner(), measurement=_measurement())


def _context(**updates: str) -> CorrelationContext:
    values = {
        "run_id": "run",
        "trace_id": "0123456789abcdef0123456789abcdef",
        "span_id": "0123456789abcdef",
        "backend": "numpy",
        "fallback": "none",
        "numerical_policy_id": "policy",
        **updates,
    }
    return CorrelationContext(**values)


def test_otlp_nested_privacy_and_any_value_branches() -> None:
    assert _contains_sensitive_json({"safe": ["token=secret"]}) is True
    assert _contains_sensitive_json(["safe", "value"]) is False
    assert _any_value(None) == {"stringValue": "null"}
    assert _any_value(True) == {"boolValue": True}
    assert _any_value(2) == {"intValue": "2"}
    assert _any_value(2.5) == {"doubleValue": 2.5}
    with pytest.raises(TelemetryContractError, match="unsupported"):
        _any_value(object())


@pytest.mark.parametrize(
    "updates",
    [
        {"run_id": ""},
        {"trace_id": "not-hex"},
        {"span_id": "not-hex"},
    ],
)
def test_otlp_correlation_rejects_empty_or_malformed_fields(
    updates: dict[str, str],
) -> None:
    with pytest.raises(TelemetryContractError):
        _context(**updates)


def test_otlp_message_and_endpoint_validation_branches() -> None:
    with pytest.raises(TelemetryContractError, match="empty"):
        build_otlp_log_request("", _context())
    for endpoint in (
        "http://127.0.0.1:4318/v1/traces",
        "http://127.0.0.1:4318/v1/logs?token=x",
        "ftp://localhost/v1/logs",
    ):
        with pytest.raises(TelemetryContractError):
            _safe_endpoint(endpoint)
    _safe_endpoint("http://localhost:4318/v1/logs")
    _safe_endpoint("https://collector.example/v1/logs")


def _wheel(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)


def test_reproducibility_rejects_unsafe_unsupported_empty_and_recordless(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe.zip"
    _wheel(unsafe, [("../escape.txt", b"x")])
    with pytest.raises(ValueError, match="unsafe"):
        normalized_archive_report(unsafe, runner="r")
    unsupported = tmp_path / "unsupported.bin"
    unsupported.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        normalized_archive_report(unsupported, runner="r")
    empty = tmp_path / "empty.zip"
    _wheel(empty, [])
    with pytest.raises(ValueError, match="no regular"):
        normalized_archive_report(empty, runner="r")
    recordless = tmp_path / "recordless.whl"
    _wheel(recordless, [("pkg/a.py", b"x")])
    with pytest.raises(ValueError, match="exactly one"):
        normalized_archive_report(recordless, runner="r")
    with pytest.raises(ValueError, match="runner"):
        normalized_archive_report(unsafe, runner=" ")


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (b"\xff", "UTF-8"),
        (b"only,two\n", "path, digest"),
        (b"pkg/a.py,sha256=x,1\npkg/a.py,sha256=x,1\n", "integrity"),
        (b"pkg-1.dist-info/RECORD,sha256=x,1\n", "hash itself"),
        (b"missing.py,sha256=x,1\npkg-1.dist-info/RECORD,,\n", "missing member"),
    ],
)
def test_wheel_record_semantics_fail_closed(
    tmp_path: Path, record: bytes, message: str
) -> None:
    wheel = tmp_path / "invalid.whl"
    _wheel(wheel, [("pkg/a.py", b"x"), ("pkg-1.dist-info/RECORD", record)])
    with pytest.raises(ValueError, match=message):
        normalized_archive_report(wheel, runner="r")


def test_record_and_archive_duplicate_or_inventory_drift_branches(tmp_path: Path) -> None:
    digest = "LXEWQrcmsEQBYnyp-6wy9chTD7GQPMTbAiWHF5IaSIE"
    duplicate_record = tmp_path / "duplicate-record.whl"
    record = (
        f"pkg/a.py,sha256={digest},1\n"
        f"pkg/a.py,sha256={digest},1\n"
        "pkg-1.dist-info/RECORD,,\n"
    ).encode()
    _wheel(
        duplicate_record,
        [("pkg/a.py", b"x"), ("pkg-1.dist-info/RECORD", record)],
    )
    with pytest.raises(ValueError, match="duplicate wheel RECORD"):
        normalized_archive_report(duplicate_record, runner="r")

    incomplete_record = tmp_path / "incomplete.whl"
    _wheel(
        incomplete_record,
        [
            ("pkg/a.py", b"x"),
            ("pkg-1.dist-info/RECORD", b"pkg-1.dist-info/RECORD,,\n"),
        ],
    )
    with pytest.raises(ValueError, match="inventory"):
        normalized_archive_report(incomplete_record, runner="r")

    duplicate_archive = tmp_path / "duplicate.zip"
    _wheel(duplicate_archive, [("pkg/a.py", b"x"), ("pkg/a.py", b"x")])
    with pytest.raises(ValueError, match="duplicate archive"):
        normalized_archive_report(duplicate_archive, runner="r")

    multiple_records = tmp_path / "multiple.zip"
    _wheel(
        multiple_records,
        [
            ("pkg/a.dist-info/RECORD", b""),
            ("pkg/b.dist-info/RECORD", b""),
        ],
    )
    with pytest.raises(ValueError, match="multiple"):
        normalized_archive_report(multiple_records, runner="r")


def test_reproducibility_handles_binary_text_and_tar_directories(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "binary.zip"
    _wheel(binary, [("pkg/a.txt", b"\xff")])
    assert normalized_archive_report(binary, runner="r")["entries"]
    source = tmp_path / "source.txt"
    source.write_text("x", encoding="utf-8")
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(tmp_path, arcname="root", recursive=False)
        output.add(source, arcname="root/source.txt")
    assert normalized_archive_report(archive, runner="r")["entries"]


def test_digest_comparison_rejects_every_report_contract_surface() -> None:
    valid: dict[str, object] = {
        "schema_version": "1.0.0",
        "normalization": "sorted-paths+declared-utf8-text-lf+content-sha256+record-semantics-v1",
        "normalized_sha256": "a" * 64,
        "entries": [{"path": "a", "sha256": "b" * 64, "size": 1}],
        "runner": "left",
    }
    for key, value in (
        ("schema_version", "0"),
        ("normalization", "other"),
        ("normalized_sha256", "short"),
        ("entries", {}),
    ):
        invalid = deepcopy(valid)
        invalid[key] = value
        with pytest.raises(ArtifactMismatch):
            compare_digest_reports(invalid, {**valid, "runner": "right"})
    with pytest.raises(ArtifactMismatch, match="runner"):
        compare_digest_reports(valid, valid)
    different = {**valid, "runner": "right", "entries": []}
    with pytest.raises(ArtifactMismatch, match="inventories"):
        compare_digest_reports(valid, different)


def test_candidate_baseline_remains_valid_json() -> None:
    json.dumps(_baseline())
