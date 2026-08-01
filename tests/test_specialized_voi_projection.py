from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vop_poc_nz import specialized_voi_projection
from vop_poc_nz.specialized_voi_projection import (
    dispatch,
    dispatch_plan,
    load_projection,
)

PROJECTION = (
    Path(__file__).parents[1]
    / "conductor/tracks/specialized-voi-v1-2_20260727/projection.json"
)
PROJECTION_V13 = (
    Path(__file__).parents[1]
    / "conductor/tracks/specialized-voi-v1-3_20260801/projection.json"
)


def test_c16_projection_yields_only_explicitly_registered_dispatch_target() -> None:
    projection = load_projection(PROJECTION)

    plan = dispatch_plan(projection, "0123456789abcdef")

    assert plan["event_type"] == "canonical-specialized-voi-updated"
    assert plan["targets"] == ["edithatogo/voiage"]
    assert plan["client_payload"]["canonical_track"] == "C16"
    assert plan["client_payload"]["canonical_ref"] == "0123456789abcdef"
    assert plan["client_payload"]["projection_path"].endswith(
        "specialized-voi-v1-2_20260727/projection.json"
    )


def test_c17_projection_records_mcda_delivery_and_explicit_versioned_path() -> None:
    projection = load_projection(PROJECTION_V13)
    projection_path = "conductor/tracks/specialized-voi-v1-3_20260801/projection.json"
    plan = dispatch_plan(projection, "fedcba9876543210", projection_path)
    issue = next(issue for issue in projection["issues"] if issue["number"] == 560)

    assert projection["projection_id"] == "specialized-voi-v1.3.0"
    assert projection["contract_version"] == "v1.3.0"
    assert projection["canonical_track"] == "C17"
    assert issue["implementation_status"] == "experimental_repository_evidence"
    assert issue["subissues"] == [746, 747, 748, 749, 750]
    assert issue["implementation_pr"] == 751
    assert issue["requirement_ids"] == ["M17", "M21"]
    assert plan["client_payload"]["projection_path"].endswith(
        "specialized-voi-v1-3_20260801/projection.json"
    )


def test_dispatch_rejects_a_projection_path_from_another_version() -> None:
    with pytest.raises(ValueError, match="projection path"):
        dispatch_plan(load_projection(PROJECTION_V13), "local-test", PROJECTION)


def test_projection_rejects_an_unregistered_projection_id(tmp_path: Path) -> None:
    value = json.loads(PROJECTION_V13.read_text(encoding="utf-8"))
    value["projection_id"] = "specialized-voi-v9.9.9"

    with pytest.raises(ValueError, match="is not registered"):
        load_projection(_write_projection(tmp_path, value))


def test_c16_projection_records_study_efficiency_repository_evidence() -> None:
    projection = load_projection(PROJECTION)
    issue = next(issue for issue in projection["issues"] if issue["number"] == 571)

    assert issue["implementation_status"] == "experimental_repository_evidence"
    assert issue["capability_contract"] == (
        "conductor/tracks/study_design_efficiency_20260727/contract.md"
    )
    assert issue["subissues"] == [680, 681, 682]
    assert issue["implementation_pr"] == 679


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


def _write_projection(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"canonical_track": "C99"}), "canonical_track"),
        (
            lambda value: value.update({"registered_repositories": []}),
            "registered_repositories",
        ),
        (
            lambda value: value.update({"registered_repositories": ["not-an-object"]}),
            "each registered repository",
        ),
        (
            lambda value: value["registered_repositories"][0].update(
                {"managed_projection": False}
            ),
            "not explicitly managed",
        ),
        (
            lambda value: value.update(
                {"registered_repositories": value["registered_repositories"] * 2}
            ),
            "must be unique",
        ),
        (lambda value: value.update({"issues": []}), "issues must be"),
        (lambda value: value.update({"issues": ["not-an-object"]}), "each issue"),
        (
            lambda value: value["issues"][0].update({"repository": ""}),
            "issue repository",
        ),
        (
            lambda value: value["issues"][0].update({"number": "318"}),
            "positive integer",
        ),
        (
            lambda value: value["issues"][0].update({"number": 0}),
            "positive integer",
        ),
        (
            lambda value: value["issues"][1].update(
                {"implementation_status": "stable"}
            ),
            "implementation_status",
        ),
        (
            lambda value: value["issues"][1].update({"capability_contract": ""}),
            "capability_contract",
        ),
        (
            lambda value: value["issues"][1].update({"subissues": [680, 680]}),
            "subissues",
        ),
        (
            lambda value: value["issues"][1].update({"implementation_pr": 0}),
            "implementation_pr",
        ),
    ],
)
def test_c16_projection_rejects_each_invalid_registration_or_issue_shape(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    value = json.loads(PROJECTION.read_text(encoding="utf-8"))
    mutate(value)

    with pytest.raises(ValueError, match=message):
        load_projection(_write_projection(tmp_path, value))


def test_c16_projection_rejects_a_non_object_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="root"):
        load_projection(_write_projection(tmp_path, []))


def test_dispatch_plan_requires_a_non_empty_canonical_ref() -> None:
    with pytest.raises(ValueError, match="canonical_ref"):
        dispatch_plan(load_projection(PROJECTION), "")


def test_dispatch_fails_closed_without_a_cross_repository_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOVERNANCE_SYNC_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="GOVERNANCE_SYNC_TOKEN"):
        dispatch(dispatch_plan(load_projection(PROJECTION), "local-test"))


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_dispatch_posts_only_the_validated_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def fake_urlopen(request: object, timeout: int) -> _Response:
        captured.extend((request, timeout))
        return _Response(204)

    monkeypatch.setattr(specialized_voi_projection, "urlopen", fake_urlopen)
    dispatch(
        dispatch_plan(load_projection(PROJECTION), "local-test"), token="test-token"
    )

    request, timeout = captured
    assert timeout == 30
    assert (
        request.full_url == "https://api.github.com/repos/edithatogo/voiage/dispatches"
    )
    assert request.get_header("Authorization") == "Bearer test-token"


def test_dispatch_uses_environment_credential_and_rejects_unexpected_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOVERNANCE_SYNC_TOKEN", "environment-token")
    monkeypatch.setattr(
        specialized_voi_projection, "urlopen", lambda _request, timeout: _Response(500)
    )

    with pytest.raises(RuntimeError, match="returned 500"):
        dispatch(dispatch_plan(load_projection(PROJECTION), "local-test"))
