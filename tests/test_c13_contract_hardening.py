"""Regression tests for C13 review findings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from vop_poc_nz.compat.legacy import intervention_spec_from_legacy, run_typed_cea
from vop_poc_nz.concerns import GitHubSyncPayload
from vop_poc_nz.domain.contracts import (
    AnalysisSpec,
    DistributionFamily,
    DistributionParameter,
    DistributionSpec,
    NumericalPolicySpec,
    ParameterSpec,
    ProvenanceSpec,
    UnitDimension,
    UnitSpec,
)
from vop_poc_nz.github_sync_planner import GitHubIssueSnapshot, plan_github_sync
from vop_poc_nz.pipeline.typed import TypedPipelineSpec, typed_pipeline_spec_from_legacy
from vop_poc_nz.results.base import ResultMaturity


def _parameters() -> dict[str, object]:
    return {
        "states": ["Healthy", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.8, 0.2], [0.0, 1.0]],
            "new_treatment": [[0.9, 0.1], [0.0, 1.0]],
        },
        "cycles": 3,
        "initial_population": [100.0, 0.0],
        "costs": {
            "health_system": {
                "standard_care": [10.0, 0.0],
                "new_treatment": [15.0, 0.0],
            },
            "societal": {
                "standard_care": [2.0, 0.0],
                "new_treatment": [-1.0, 0.0],
            },
        },
        "qalys": {
            "standard_care": [1.0, 0.0],
            "new_treatment": [1.0, 0.0],
        },
    }


@pytest.mark.parametrize("cycles", [True, 3.9, "3"])
def test_legacy_cycles_reject_lossy_or_ambiguous_values(cycles: object) -> None:
    parameters = _parameters()
    parameters["cycles"] = cycles
    with pytest.raises(TypeError, match="cycles must be an integer"):
        intervention_spec_from_legacy(parameters)


def test_legacy_numeric_strings_and_boolean_wtp_are_rejected() -> None:
    parameters = _parameters()
    parameters["discount_rate"] = "0.03"
    with pytest.raises(TypeError, match="discount_rate must be a real number"):
        intervention_spec_from_legacy(parameters)
    with pytest.raises(TypeError, match="wtp_threshold must be a real number"):
        run_typed_cea(_parameters(), wtp_threshold=True)  # type: ignore[arg-type]


def test_pipeline_rejects_forged_fingerprint_and_duplicate_names() -> None:
    valid = typed_pipeline_spec_from_legacy(
        {"A": _parameters()},
        run_id="run",
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    payload = valid.model_dump()
    with pytest.raises(ValidationError, match="spec_fingerprint does not match"):
        TypedPipelineSpec(**{**payload, "spec_fingerprint": "0" * 64})
    with pytest.raises(ValidationError, match="intervention names must be unique"):
        TypedPipelineSpec(**{**payload, "interventions": (valid.interventions[0],) * 2})


def test_typed_contracts_carry_units_policy_provenance_and_result_identity() -> None:
    provenance = ProvenanceSpec(source_id="fixture:test")
    currency = UnitSpec(
        symbol="NZD",
        dimension=UnitDimension.CURRENCY,
        currency_code="NZD",
        currency_year=2026,
    )
    parameter = ParameterSpec(
        name="cost",
        value=12.0,
        unit=currency,
        dimensions=("strategy",),
        provenance=provenance,
    )
    analysis = AnalysisSpec(
        analysis_type="cea",
        contract_version="1.0.0",
        parameters=(parameter,),
        numerical_policy=NumericalPolicySpec(relative_tolerance=1e-8),
    )
    distribution = DistributionSpec(
        family=DistributionFamily.GAMMA,
        parameters=(
            DistributionParameter(name="shape", value=2.0),
            DistributionParameter(name="scale", value=4.0),
        ),
        unit=currency,
        provenance=provenance,
    )
    result = run_typed_cea(_parameters())

    assert analysis.parameters[0].unit.currency_year == 2026
    assert distribution.family is DistributionFamily.GAMMA
    assert result.metadata.maturity is ResultMaturity.STABLE
    assert len(result.metadata.arrow_schema.schema_fingerprint) == 64


def test_typed_kernel_emits_no_legacy_logs_or_warnings(caplog) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_typed_cea(_parameters())
    assert caught == []
    assert caplog.records == []


def test_sync_preserves_human_metadata_present_in_base() -> None:
    marker = "vop-voiage-governance-id:CON-SHR-0013"
    base_body = (
        f"<!-- {marker} -->\nhuman\n<!-- governance:begin -->old<!-- governance:end -->"
    )
    local = GitHubSyncPayload(
        github_repository="owner/repository",
        issue_number=1,
        desired_state="open",
        stable_marker=marker,
        title="title",
        body=base_body.replace("old", "new"),
        labels=("managed",),
        project_number=1,
        project_fields=(("Managed", "new"),),
    )
    base = GitHubIssueSnapshot(
        github_repository=local.github_repository,
        issue_number=1,
        state="open",
        title="title",
        body=base_body,
        labels=("managed", "human-old"),
        project_number=1,
        project_fields=(("Managed", "old"), ("Human", "keep")),
        managed_labels=("managed",),
        managed_project_field_names=("Managed",),
    )
    plan = plan_github_sync(base=base, local=local, remote=replace(base))

    assert plan.outcome == "local_only"
    assert plan.proposed_issue is not None
    assert plan.proposed_issue.labels == ("human-old", "managed")
    assert plan.proposed_issue.project_fields == (
        ("Human", "keep"),
        ("Managed", "new"),
    )


def test_dsa_compatibility_import_is_side_effect_free(tmp_path) -> None:
    code = """
import contextlib
import importlib
import io
import json
from pathlib import Path
stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    module = importlib.import_module('src.test_dsa_enhancements')
print(json.dumps({
    'stdout': stdout.getvalue(),
    'stderr': stderr.getvalue(),
    'entries': sorted(path.name for path in Path.cwd().iterdir()),
    'has_main': callable(module.main),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "PYTHONWARNINGS": "ignore::DeprecationWarning",
        },
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"entries": [], "has_main": True, "stderr": "", "stdout": ""}
