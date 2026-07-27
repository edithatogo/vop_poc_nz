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
