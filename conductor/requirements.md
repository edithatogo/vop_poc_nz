# VOP–VOIAGE requirements

This document is the requirements baseline for the cross-repository Conductor
programme. `conductor/manifest.json` remains the canonical track registry and
the linked track files contain detailed acceptance criteria.

## MoSCoW priorities

### Must have

- **M1 — Map first (C00):** inspect both live worktrees, Git history, private
  boundaries, and existing implementations before mutation.
- **M2 — Stable method contract (C01–C02):** directional current-information
  EVoP and the Perspective Acceptability Frontier have versioned semantics,
  explicit tie handling, deterministic fixtures, and cross-package tests.
- **M3 — Evidence truth (C03–C04):** every policy-facing input and claim is
  traceable to a source or derivation; synthetic, tutorial, validation, and
  policy-grade cases remain distinguishable.
- **M4 — Reproducible artifacts (C05):** generated outputs have deterministic
  manifests, schema fingerprints, provenance, and fail-closed promotion rules.
- **M5 — Publication governance (C06):** manuscript, release, registry, and
  public-deployment claims require their own evidence and human authorization.
- **M6 — Maintained repositories (C07–C08):** VOP and VOIAGE use supported
  Python, locked current dependencies, quality gates, and documented APIs.
- **M7 — Explicit integration (C09):** repositories interoperate through the
  versioned VOP–VOIAGE contract, Arrow IPC/Parquet, and adapters rather than
  source-tree imports or repository merging.
- **M8 — Supply-chain truth (C10):** CI, releases, documentation, dependency
  audits, and software bills of materials describe the code actually shipped.
- **M9 — Measured performance (C11):** optimization decisions use reproducible
  benchmarks, numerical parity checks, and declared fallback behavior.
- **M10 — Durable roadmap (C12):** every current track has one identifiable
  GitHub issue; historical tracks and development eras are preserved in closed
  ledger issues; all are represented in the GitHub Project.
- **M11 — Privacy boundary:** `.conductor/local`, credentials, unpublished
  evidence, and owner-only decisions are never promoted implicitly.

### Should have

- **S1:** PyArrow and Polars round trips preserve logical types, metadata, and
  values across processes and operating systems.
- **S2:** supported Python releases include 3.14; free-threaded Python is
  monitored as an observational, wheel-only frontier until dependencies mature.
- **S3:** lint, dependency, coverage, evidence, and performance debt use
  ratchets so the baseline cannot silently regress.
- **S4:** project items expose repository, track ID, MoSCoW priority, lifecycle
  status, and whether a remaining gate is local, external, or human-controlled.
- **S5:** closed historical ledgers retain links to the original Conductor
  registry, archived track paths, releases, commits, and pull requests.

### Could have

- **C1:** native Rust or other-language implementations consuming the same
  compatibility contract and golden fixtures.
- **C2:** GPU-backed Polars or Arrow acceleration where parity and benchmark
  evidence justify an experimental lane.
- **C3:** automated project-field synchronization from the local track registry.
- **C4:** richer model cards and case-level provenance visualizations.
- **C5:** signed provenance and attestations for promoted interchange bundles.

### Won't have now

- **W1:** automatic publication, registry submission, manuscript submission, or
  policy claims without explicit owner approval.
- **W2:** direct VOP-to-VOIAGE source imports or a forced monorepo migration.
- **W3:** production claims for free-threaded Python, GPU, TPU, FPGA, ASIC, or
  external registries based only on local fixtures or unavailable hardware.
- **W4:** implicit publication of private/local Conductor state.
- **W5:** retroactive rewriting or deletion of historical tracks merely because
  they are superseded.

## Verification

Requirements are evidenced by the track acceptance criteria, deterministic
fixtures, repository harnesses, hosted CI, issue/project reconciliation, and
explicit external-gate records. A green local test never substitutes for a
publication, evidence, credential, reviewer, or governance gate.
