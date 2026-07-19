"""Deterministic private-draft staging for the standalone contract bundle."""

from __future__ import annotations

import gzip
import json
import re
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from .contract_bundle import verify_contract_bundle

_TAG_RE = re.compile(
    r"^vop-voiage-contracts-v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9A-Za-z.-]+)?)$"
)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_bytes(_canonical_bytes(value))
    return _sha256(path)


def _archive_bundle(bundle_root: Path, output: Path, *, version: str) -> None:
    prefix = f"vop-voiage-contracts-{version}"
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for path in sorted(bundle_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(bundle_root).as_posix()
            info = archive.gettarinfo(str(path), arcname=f"{prefix}/{relative}")
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as source:
                archive.addfile(info, source)


def _cyclonedx_sbom(manifest: dict[str, Any]) -> dict[str, object]:
    components: list[dict[str, object]] = []
    for entry in cast("list[dict[str, object]]", manifest["files"]):
        components.append(
            {
                "type": "file",
                "name": entry["path"],
                "hashes": [{"alg": "SHA-256", "content": entry["sha256"]}],
                "properties": [
                    {"name": "vop.media_type", "value": entry["media_type"]},
                    {"name": "vop.size_bytes", "value": str(entry["size"])},
                ],
            }
        )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "data",
                "name": manifest["bundle_id"],
                "version": manifest["bundle_version"],
                "hashes": [
                    {
                        "alg": "SHA-256",
                        "content": manifest["bundle_sha256"],
                    }
                ],
            }
        },
        "components": components,
    }


def stage_contract_bundle(
    bundle_root: Path,
    output_dir: Path,
    *,
    release_tag: str,
    source_revision: str,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Stage exact bytes and evidence without authorizing publication."""
    matched = _TAG_RE.fullmatch(release_tag)
    if matched is None:
        raise ValueError("release tag must use vop-voiage-contracts-v<version>")
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise ValueError("source revision must be an exact lowercase Git commit SHA")
    if _SHA256_RE.fullmatch(expected_bundle_sha256) is None:
        raise ValueError("expected bundle digest must be a lowercase SHA-256")
    manifest = verify_contract_bundle(bundle_root)
    bundle_sha256 = manifest.get("bundle_sha256")
    if bundle_sha256 != expected_bundle_sha256:
        raise ValueError("expected bundle digest does not match the verified manifest")
    version = manifest.get("bundle_version")
    if matched["version"] != version:
        raise ValueError("release tag version does not match the bundle version")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("staging output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_archive = output_dir / "bundle.tar.gz"
    _archive_bundle(bundle_root, temporary_archive, version=str(version))
    artifact_sha256 = _sha256(temporary_archive)
    artifact_name = (
        f"vop-voiage-contracts-{version}-{expected_bundle_sha256[:16]}-"
        f"{artifact_sha256}.tar.gz"
    )
    artifact = temporary_archive.with_name(artifact_name)
    temporary_archive.replace(artifact)

    sbom_name = "sbom.cdx.json"
    sbom_path = output_dir / sbom_name
    sbom_sha256 = _write_json(sbom_path, _cyclonedx_sbom(manifest))

    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": artifact_name, "digest": {"sha256": artifact_sha256}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/edithatogo/vop_poc_nz/contract-bundle-stage/v1",
                "externalParameters": {
                    "release_tag": release_tag,
                    "source_revision": source_revision,
                    "bundle_sha256": expected_bundle_sha256,
                },
                "resolvedDependencies": [
                    {
                        "uri": "git+https://github.com/edithatogo/vop_poc_nz",
                        "digest": {"gitCommit": source_revision},
                    },
                    {
                        "uri": f"contracts/vop-voiage/{version}/manifest.json",
                        "digest": {"sha256": _sha256(bundle_root / "manifest.json")},
                    },
                ],
            },
            "runDetails": {
                "builder": {
                    "id": "https://github.com/edithatogo/vop_poc_nz/.github/workflows/contract-bundle-stage.yml"
                },
                "metadata": {"invocationId": release_tag},
            },
        },
    }
    provenance_name = "provenance.json"
    provenance_path = output_dir / provenance_name
    provenance_sha256 = _write_json(provenance_path, provenance)
    predicate_name = "provenance-predicate.json"
    predicate_path = output_dir / predicate_name
    predicate_sha256 = _write_json(predicate_path, provenance["predicate"])

    stage = {
        "schema_version": "1.0.0",
        "kind": "standalone_contract_bundle_stage",
        "publication_state": "private_draft_staging_only",
        "source": {
            "repository": "edithatogo/vop_poc_nz",
            "revision": source_revision,
            "tag": release_tag,
        },
        "bundle": {
            "bundle_id": manifest["bundle_id"],
            "bundle_version": version,
            "bundle_sha256": expected_bundle_sha256,
            "manifest_sha256": _sha256(bundle_root / "manifest.json"),
        },
        "artifact": {
            "path": artifact_name,
            "sha256": artifact_sha256,
            "media_type": "application/gzip",
        },
        "evidence": {
            "sbom": {"path": sbom_name, "sha256": sbom_sha256},
            "provenance": {
                "path": provenance_name,
                "sha256": provenance_sha256,
                "predicate_path": predicate_name,
                "predicate_sha256": predicate_sha256,
            },
        },
        "authorization": {
            "publication_authorized": False,
            "signing_authorized": False,
            "requires_human_release_approval": True,
        },
        "network_mutation": False,
    }
    _write_json(output_dir / "stage-manifest.json", stage)
    return stage


__all__ = ["stage_contract_bundle"]
