"""Standalone content-addressed contract bundle staging tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vop_poc_nz.contract_bundle_staging import stage_contract_bundle

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "contracts/vop-voiage/1.0.0"
BUNDLE_SHA256 = "f79a8d56b22736e34f10d8cb02db46239f27093fd1c366d4ca0ba2c688b60798"
SOURCE_REVISION = "b" * 40
TAG_OID = "d" * 40


def _tag_verification(tmp_path: Path) -> Path:
    path = tmp_path / "tag-verification-input.json"
    path.write_text(
        json.dumps(
            {
                "tag": "vop-voiage-contracts-v1.0.0",
                "object": {"type": "commit", "sha": SOURCE_REVISION},
                "verification": {"verified": True, "reason": "valid"},
                "tagger": {"email": "release@example.test"},
            }
        )
    )
    return path


def test_staging_is_deterministic_content_addressed_and_complete(
    tmp_path: Path,
) -> None:
    tag_verification = _tag_verification(tmp_path)
    first = stage_contract_bundle(
        BUNDLE,
        tmp_path / "first",
        release_tag="vop-voiage-contracts-v1.0.0",
        source_revision=SOURCE_REVISION,
        expected_bundle_sha256=BUNDLE_SHA256,
        tag_object_oid=TAG_OID,
        tag_target_commit=SOURCE_REVISION,
        tag_verification_path=tag_verification,
    )
    second = stage_contract_bundle(
        BUNDLE,
        tmp_path / "second",
        release_tag="vop-voiage-contracts-v1.0.0",
        source_revision=SOURCE_REVISION,
        expected_bundle_sha256=BUNDLE_SHA256,
        tag_object_oid=TAG_OID,
        tag_target_commit=SOURCE_REVISION,
        tag_verification_path=tag_verification,
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
    assert first["source"]["tag_object_oid"] == TAG_OID
    assert first["source"]["tag_target_commit"] == SOURCE_REVISION
    assert {item["path"] for item in first["release_assets"]} == {
        path.name for path in (tmp_path / "first").iterdir()
    } - {"stage-manifest.json"}

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
            tag_object_oid=TAG_OID,
            tag_target_commit=SOURCE_REVISION,
            tag_verification_path=_tag_verification(tmp_path),
        )


def test_staging_rejects_unverified_or_mismatched_tag_evidence(tmp_path: Path) -> None:
    evidence = _tag_verification(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["object"]["sha"] = "e" * 40
    evidence.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exact tag"):
        stage_contract_bundle(
            BUNDLE,
            tmp_path / "stage",
            release_tag="vop-voiage-contracts-v1.0.0",
            source_revision=SOURCE_REVISION,
            expected_bundle_sha256=BUNDLE_SHA256,
            tag_object_oid=TAG_OID,
            tag_target_commit=SOURCE_REVISION,
            tag_verification_path=evidence,
        )


def test_dedicated_workflows_are_dispatch_only_and_fail_closed() -> None:
    capture = (ROOT / ".github/workflows/governance-baseline-capture.yml").read_text()
    governance = (ROOT / ".github/workflows/governance-baseline-review.yml").read_text()
    bundle = (ROOT / ".github/workflows/contract-bundle-stage.yml").read_text()

    assert "workflow_dispatch:" in capture
    assert "pull_request:" not in capture
    assert "permissions:\n  contents: read\n  issues: read" in capture
    assert "ref: ${{ inputs.source_revision }}" not in capture
    assert 'git show "$SOURCE_REVISION:.github/governance-baselines/' in capture
    assert "git merge-base --is-ancestor" in capture
    assert "workflow_dispatch:" in governance
    assert "pull_request:" not in governance
    assert "environment: governance-baseline-approval" in governance
    assert "ref: ${{ inputs.source_revision }}" not in governance
    assert "/approvals" in governance
    assert "capture-run.json" in governance
    assert "candidate_sha256" in governance
    assert "--approval-history" in governance

    assert "workflow_dispatch:" in bundle
    assert "push:" not in bundle
    assert "private draft" in bundle.lower()
    assert "contents: write" in bundle
    assert "attestations: write" in bundle
    assert "id-token: write" in bundle
    assert "git cat-file -t" in bundle
    assert "verification.verified == true" in bundle
    assert "publication_authorized" in bundle
    assert "git/ref/tags/$RELEASE_TAG" in bundle
    assert "comm -23 existing-assets.txt expected-assets.txt" in bundle
    assert "bundle-stage/* --clobber" not in bundle
    assert "group: contract-bundle-private-draft-${{ inputs.release_tag }}" in bundle
    assert "release_id=\"$(jq -r '.id' release-before.json)\"" in bundle
    assert "releases/${release_id}/assets?name=${asset}" in bundle
    assert "repos/${GITHUB_REPOSITORY}/releases/$release_id" in bundle
    assert ".draft == true and .tag_name == $tag" in bundle
    assert 'test("^sha256:[0-9a-f]{64}$")' in bundle
    assert "diff -u expected-assets.tsv uploaded-assets.tsv" in bundle
    assert bundle.count("git/ref/tags/$RELEASE_TAG") == 2
