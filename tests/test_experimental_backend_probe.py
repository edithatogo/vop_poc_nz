"""Tests for the opt-in JAX backend evidence probe."""

import pytest

pytest.importorskip("jax")

from scripts.experimental_backend_probe import build_evidence


def test_experimental_backend_probe_executes_jitted_x64_calculation() -> None:
    evidence = build_evidence()

    assert evidence["backend"] in {"cpu", "gpu", "tpu"}
    assert evidence["device_kind"]
    assert evidence["jax_version"]
    assert evidence["result"] == 14.0
    assert evidence["x64_enabled"] is True
