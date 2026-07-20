# Arrow interchange and evidence harness

The public interchange boundary is Apache Arrow. Parquet is the durable tabular
format, Arrow IPC is the versioned conformance-fixture format, and the Arrow C
Stream PyCapsule is the in-process zero-copy boundary. JSON Lines remains an
explicit debugging format.

The cross-repository compatibility policy is machine-readable at
`contracts/vop-voiage/compatibility/v1/contract.json`. VOP owns this C09
contract; VOIAGE consumes a byte-identical mirror pinned to a VOP commit and
digest. The fingerprint is SHA-256 over canonical JSON for the ordered logical
field model, rather than Arrow's implementation-specific binary schema message.

## Conformance fixtures

`tests/fixtures/interchange/v1/` contains small synthetic Parquet and Arrow IPC
golden files plus `contract.json`. The contract records the logical Arrow schema
fingerprint. Metadata is deliberately excluded from the fingerprint so PyArrow,
Polars, Parquet, and IPC share the same logical identity.

Regenerate the fixtures only when intentionally versioning the interchange
contract:

```powershell
uv run python scripts/generate_interchange_fixtures.py
uv run pytest tests/test_perspective_io.py
```

Never replace an existing version in a released branch. Add `v2/` and retain
the previous fixture so new releases prove backwards readability.

## Evidence manifests

The evidence-manifest CLI hashes repository-relative paths in sorted order.
UTF-8 text is normalised to LF before hashing, which produces identical
digests on Windows and Linux. Binary content is hashed byte-for-byte.

```powershell
uv run python scripts/evidence_manifest.py generate evidence.manifest.json output/table.parquet --root .
uv run python scripts/evidence_manifest.py verify evidence.manifest.json --root .
```

A manifest does not promote a file across the publication boundary. Artifact
classification and Conductor publication gates still apply.

## Quality and performance ratchets

`scripts/lint_ratchet.py` rejects any new Ruff rule or increase above the
checked-in per-rule baseline. Existing debt can therefore be reduced in small
changes without a broad rewrite. Update the baseline only when counts decrease.

`scripts/serialization_benchmark.py` compares Arrow write times with JSON Lines
using ratios measured on the same host. The generous initial budgets avoid
hardware-specific absolute timing gates; tighten them after enough hosted runs
have established a stable distribution.

Python 3.14t runs as a non-blocking CI observation while free-threaded support
remains experimental. A failure is evidence to investigate, not permission to
weaken the supported Python 3.14 gate.
