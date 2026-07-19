"""Standalone content-addressed contract bundle staging tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vop_poc_nz.contract_bundle_staging import stage_contract_bundle

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "contracts/vop-voiage/1.0.0"
BUNDLE_SHA256 = "f79a8d56b22736e34f10d8cb02db46239f27093fd1c366d4ca0ba2c688b60798"


def test_staging_is_deterministic_content_addressed_and_complete(
    tmp_path: Path,
) -> None:
    first = stage_contract_bundle(
        BUNDLE,
        tmp_path / "first",
        release_tag="vop-voiage-contracts-v1.0.0",
        source_revision="b" * 40,
        expected_bundle_sha256=BUNDLE_SHA256,
    )
    second = stage_contract_bundle(
        BUNDLE,
        tmp_path / "second",
        release_tag="vop-voiage-contracts-v1.0.0",
        source_revision="b" * 40,
        expected_bundle_sha256=BUNDLE_SHA256,
    )

    first_files = {
        path.name: path.read_bytes() for path in (tmp_path / "first").iterdir()
    }
    second_files = {
        path.name: path.read_bytes() for path in (tmp_path / "second").iterdir()
    }
    assert first_files == second_files
    assert first == second
    assert first["publication_state"] == "private_draft_staging_only"
    assert first["authorization"]["publication_authorized"] is False
    assert first["bundle"]["bundle_sha256"] == BUNDLE_SHA256
    assert BUNDLE_SHA256[:16] in first["artifact"]["path"]
    assert set(first["evidence"]) == {"sbom", "provenance"}

    sbom = json.loads(
        (tmp_path / "first" / first["evidence"]["sbom"]["path"]).read_text()
    )
    provenance = json.loads(
        (tmp_path / "first" / first["evidence"]["provenance"]["path"]).read_text()
    )
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert provenance["predicateType"] == "https://slsa.dev/provenance/v1"
    assert provenance["subject"][0]["digest"]["sha256"] == first["artifact"]["sha256"]


@pytest.mark.parametrize(
    ("tag", "revision", "digest", "match"),
    [
        ("v1.0.0", "b" * 40, BUNDLE_SHA256, "tag"),
        ("vop-voiage-contracts-v1.0.0", "main", BUNDLE_SHA256, "revision"),
        ("vop-voiage-contracts-v1.0.0", "b" * 40, "0" * 64, "digest"),
    ],
)
def test_staging_rejects_unbound_tag_revision_or_digest(
    tmp_path: Path, tag: str, revision: str, digest: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        stage_contract_bundle(
            BUNDLE,
            tmp_path / "stage",
            release_tag=tag,
            source_revision=revision,
            expected_bundle_sha256=digest,
        )


def test_dedicated_workflows_are_dispatch_only_and_fail_closed() -> None:
    capture = (ROOT / ".github/workflows/governance-baseline-capture.yml").read_text()
    governance = (ROOT / ".github/workflows/governance-baseline-review.yml").read_text()
    bundle = (ROOT / ".github/workflows/contract-bundle-stage.yml").read_text()

    assert "workflow_dispatch:" in capture
    assert "pull_request:" not in capture
    assert "permissions:\n  contents: read\n  issues: read" in capture
    assert "workflow_dispatch:" in governance
    assert "pull_request:" not in governance
    assert "environment: governance-baseline-approval" in governance
    assert "candidate_sha256" in governance
    assert "--approved-by" in governance

    assert "workflow_dispatch:" in bundle
    assert "push:" not in bundle
    assert "private draft" in bundle.lower()
    assert "contents: write" in bundle
    assert "attestations: write" in bundle
    assert "id-token: write" in bundle
    assert "git cat-file -t" in bundle
    assert "verification.verified == true" in bundle
    assert "publication_authorized" in bundle
