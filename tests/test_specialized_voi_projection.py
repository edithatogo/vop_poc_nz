from __future__ import annotations

import json
from pathlib import Path

import pytest

from vop_poc_nz.specialized_voi_projection import (
    dispatch,
    dispatch_plan,
    load_projection,
)

PROJECTION = (
    Path(__file__).parents[1]
    / "conductor/tracks/specialized-voi-v1-2_20260727/projection.json"
)


def test_c16_projection_yields_only_explicitly_registered_dispatch_target() -> None:
    projection = load_projection(PROJECTION)

    plan = dispatch_plan(projection, "0123456789abcdef")

    assert plan["event_type"] == "canonical-specialized-voi-updated"
    assert plan["targets"] == ["edithatogo/voiage"]
    assert plan["client_payload"]["canonical_track"] == "C16"
    assert plan["client_payload"]["canonical_ref"] == "0123456789abcdef"


def test_c16_projection_rejects_unregistered_issue_repository(tmp_path: Path) -> None:
    value = json.loads(PROJECTION.read_text(encoding="utf-8"))
    value["issues"][0]["repository"] = "edithatogo/unregistered"
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly registered"):
        load_projection(path)


def test_c16_projection_rejects_unsafe_sync_policy(tmp_path: Path) -> None:
    value = json.loads(PROJECTION.read_text(encoding="utf-8"))
    value["sync_policy"]["automatic_merge"] = True
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="fail-closed boundary"):
        load_projection(path)


def test_dispatch_fails_closed_without_a_cross_repository_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOVERNANCE_SYNC_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="GOVERNANCE_SYNC_TOKEN"):
        dispatch(dispatch_plan(load_projection(PROJECTION), "local-test"))
