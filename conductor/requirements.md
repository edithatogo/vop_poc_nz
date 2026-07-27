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
- **M12 — Observable, reproducible engineering:** package-owned logging is
  context-aware and machine-readable; versions derive from Git release tags;
  uv and Pixi expose equivalent locked commands; Pydantic v2 validates runtime
  configuration; Ruff, BasedPyright, `ty`, tests, builds, and security checks
  are visible CI contracts.
- **M13 — Typed governance and domain contracts (C13):** every material
  concern, assumption, risk, decision, evidence reference, issue link,
  parameter, numerical policy, calculation and result has a stable typed
  identity, validated relationships, provenance and a compatibility policy.
- **M14 — Estimation-focused variance VOI (C16, planned v1.2.0):**
  `EVPPI_var` and `EVSI_var` declare scalar/vector target shape, component
  units, variance or covariance functional, conditioning and sampling models,
  estimator uncertainty, diagnostics and provenance without aliasing
  decision-focused EVPPI/EVSI.
- **M15 — Study-design efficiency (C16, planned v1.2.0):** COSS returns
  evaluated designs, feasible range/set, EVSI/cost/signed-ENBS curves,
  deterministic tie policy, optimum, boundary state, uncertainty around the
  optimum and plotting inputs; EVSI/EVPI has explicit common-unit,
  zero-denominator and bounds behavior.
- **M16 — Utility and clairvoyance semantics (C16, planned v1.2.0):** Expected
  Value of Clairvoyance is an alias or presentation of the expected-utility
  value of a clairvoyant policy governed by VOIAGE issue #595, not a duplicate
  kernel or an unconditional alias for monetary EVPI.
- **M17 — Cross-repository projection (C16, planned v1.2.0):** one canonical,
  versioned public projection drives bounded managed sections, native
  issue/subissue links and Project 28 fields for every registered consumer
  repository, with dry-run, conflict detection and fail-closed credential
  handling.

### Should have

- **S1:** PyArrow and Polars round trips preserve logical types, metadata, and
  values across processes and operating systems.
- **S2:** supported Python releases include 3.12–3.14; free-threaded Python is
  monitored as an observational, wheel-only frontier until dependencies mature.
- **S3:** lint, dependency, coverage, evidence, and performance debt use
  ratchets so the baseline cannot silently regress.
- **S4:** project items expose repository, track ID, MoSCoW priority, lifecycle
  status, and whether a remaining gate is local, external, or human-controlled.
- **S5:** closed historical ledgers retain links to the original Conductor
  registry, archived track paths, releases, commits, and pull requests.
- **S6:** Scalene, mutation, dependency-audit, and experimental-backend evidence
  runs on bounded scheduled/manual lanes with artifacts, not hidden local steps.
- **S7:** GitHub synchronization uses stable markers, bounded managed sections,
  dry-run plans and three-way conflict detection while preserving human content.
- **S8:** Rust/Python/R/Julia/Mojo dispositions, accessible plots and
  independently reproducible analytical/enumerable references should be
  available before the specialized v1.2.0 contracts are promoted.

### Could have

- **C1:** native Rust or other-language implementations consuming the same
  compatibility contract and golden fixtures.
- **C2:** GPU-backed Polars or Arrow acceleration where parity and benchmark
  evidence justify an experimental lane.
- **C3:** automated project-field synchronization from the local track registry.
- **C4:** richer model cards and case-level provenance visualizations.
- **C5:** signed provenance and attestations for promoted interchange bundles.
- **C6:** deterministic generated governance tables, JSON projections and
  Mermaid traceability graphs derived from the canonical typed ledger.
- **C7:** reviewed vector-target covariance functionals beyond trace,
  determinant and declared weighted quadratic forms.

### Won't have now

- **W1:** automatic publication, registry submission, manuscript submission, or
  policy claims without explicit owner approval.
- **W2:** direct VOP-to-VOIAGE source imports or a forced monorepo migration.
- **W3:** production claims for free-threaded Python, GPU, TPU, FPGA, ASIC, or
  external registries based only on local fixtures or unavailable hardware.
- **W4:** implicit publication of private/local Conductor state.
- **W5:** retroactive rewriting or deletion of historical tracks merely because
  they are superseded.
- **W6:** automation accepting risk, approving irreversible decisions, closing
  human-controlled issues, or publishing local/private evidence.
- **W7:** a duplicate VoC numerical method, silent COSS extrapolation outside
  the feasible design set, or relabeling `total_voi / total_cost` as EVSI/EVPI.

## Planned-version traceability

| Planned version | MoSCoW | Requirements | Owning track | GitHub records |
|---|---|---|---|---|
| v1.2.0 | Must | M14 | C16 / `estimation_focused_variance_voi_20260727` | VOIAGE #619, parent #318 |
| v1.2.0 | Must | M15 | C16 / `study_design_efficiency_20260727` | VOIAGE #571, parent #318 |
| v1.2.0 | Must | M16 | C16 / `supported_frontier_method_completion_20260723` | VOIAGE #595, parent #318 |
| v1.2.0 | Must | M17 | C16 / C12 | Project 28 and every registered consumer repository |

## Verification

Requirements are evidenced by the track acceptance criteria, deterministic
fixtures, repository harnesses, hosted CI, issue/project reconciliation, and
explicit external-gate records. A green local test never substitutes for a
publication, evidence, credential, reviewer, or governance gate.
