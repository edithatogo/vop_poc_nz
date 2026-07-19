"""Reproducibility manifests for generated model results.

The intent is to prevent "orphan" manuscript results: every table, figure, or
exported dataset should be traceable to a script, input files, hashes, runtime
metadata, package version, and random seed. This module is deliberately small and
standard-library only so it can be adopted before the rest of the stack is
modernised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping


class ManifestError(ValueError):
    """Raised when a reproducibility manifest is invalid."""


@dataclass(frozen=True)
class FileHash:
    """Path and SHA-256 digest for one file."""

    path: str
    sha256: str
    role: str = "input"
    bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResultArtifact:
    """One generated result artifact, such as a figure or table."""

    artifact_id: str
    path: str
    artifact_type: str
    sha256: str
    bytes: int | None = None
    derived_from: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["derived_from"] = list(self.derived_from)
        return out


@dataclass(frozen=True)
class ResultManifest:
    """Machine-readable provenance for one analysis run."""

    run_id: str
    created_at_utc: str
    script: str
    command: str | None = None
    case_id: str | None = None
    software_version: str | None = None
    git_commit: str | None = None
    random_seed: int | None = None
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    inputs: tuple[FileHash, ...] = ()
    outputs: tuple[ResultArtifact, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def validate(self) -> None:
        """Validate minimal manuscript-reproducibility requirements."""
        if not self.run_id:
            raise ManifestError("run_id is required.")
        if not self.script:
            raise ManifestError("script is required.")
        if not self.outputs:
            raise ManifestError("At least one output artifact is required.")
        output_ids = [output.artifact_id for output in self.outputs]
        if len(set(output_ids)) != len(output_ids):
            raise ManifestError("Output artifact IDs must be unique.")
        for output in self.outputs:
            if not output.sha256:
                raise ManifestError(f"Output {output.artifact_id!r} has no hash.")
        for file_hash in self.inputs:
            if not file_hash.sha256:
                raise ManifestError(f"Input {file_hash.path!r} has no hash.")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "script": self.script,
            "command": self.command,
            "case_id": self.case_id,
            "software_version": self.software_version,
            "git_commit": self.git_commit,
            "random_seed": self.random_seed,
            "python_version": self.python_version,
            "platform": self.platform,
            "inputs": [item.as_dict() for item in self.inputs],
            "outputs": [item.as_dict() for item in self.outputs],
            "parameters": dict(self.parameters),
            "notes": self.notes,
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the manifest as deterministic pretty JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        return target


def sha256_file(path: str | Path) -> FileHash:
    """Return SHA-256 metadata for a file."""
    file_path = Path(path)
    digest = sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return FileHash(
        path=str(file_path),
        sha256=digest.hexdigest(),
        bytes=file_path.stat().st_size,
    )


def artifact_hash(
    path: str | Path,
    *,
    artifact_id: str | None = None,
    artifact_type: str | None = None,
    derived_from: Iterable[str] = (),
) -> ResultArtifact:
    """Return artifact metadata for a generated output file."""
    file_path = Path(path)
    hashed = sha256_file(file_path)
    return ResultArtifact(
        artifact_id=artifact_id or file_path.stem,
        path=hashed.path,
        artifact_type=artifact_type or file_path.suffix.lstrip(".") or "file",
        sha256=hashed.sha256,
        bytes=hashed.bytes,
        derived_from=tuple(derived_from),
    )


def build_result_manifest(
    *,
    run_id: str,
    script: str | Path,
    outputs: Iterable[str | Path],
    inputs: Iterable[str | Path] = (),
    command: str | None = None,
    case_id: str | None = None,
    software_version: str | None = None,
    git_commit: str | None = None,
    random_seed: int | None = None,
    parameters: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> ResultManifest:
    """Build a `ResultManifest` from concrete input and output paths."""
    input_hashes = tuple(sha256_file(path) for path in inputs)
    output_hashes = tuple(artifact_hash(path) for path in outputs)
    manifest = ResultManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        script=str(script),
        command=command,
        case_id=case_id,
        software_version=software_version,
        git_commit=git_commit,
        random_seed=random_seed,
        inputs=input_hashes,
        outputs=output_hashes,
        parameters=dict(parameters or {}),
        notes=notes,
    )
    manifest.validate()
    return manifest
