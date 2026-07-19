from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.c15_checkout_evidence import checkout_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_checkout_evidence_binds_expected_and_tested_source_head() -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    report = checkout_evidence(
        ROOT, expected_source_sha=head, event_name="pull_request", runner="Linux-X64"
    )

    assert report["exact_source_checkout"] is True
    assert report["expected_source_sha"] == report["tested_sha"] == head


def test_checkout_evidence_exposes_mismatch_and_rejects_invalid_sha() -> None:
    report = checkout_evidence(
        ROOT,
        expected_source_sha="0" * 40,
        event_name="workflow_dispatch",
        runner="Windows-X64",
    )
    assert report["exact_source_checkout"] is False
    with pytest.raises(ValueError, match="lowercase 40-character hex"):
        checkout_evidence(
            ROOT,
            expected_source_sha="not-a-sha",
            event_name="push",
            runner="Linux-X64",
        )
