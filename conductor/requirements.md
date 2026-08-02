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
- **M16 — Utility-equivalent information pricing and clairvoyance semantics
  (C16, planned v1.2.0):** VOIAGE issue #595 represents named utility,
  wealth/reference state, risk attitude, payoff units, information/cost
  location, current and informed policies, stakeholder scope, EUI, CEI, BPI,
  SPI, anchored PPI, policy switches, root diagnostics and cross-problem
  comparability. Expected Value of Clairvoyance is a presentation of the same
  clairvoyant-policy result, not a duplicate numerical method; monetary EVPI
  reduction requires verified positive-affine utility.
- **M17 — Cross-repository projection (C16, planned v1.2.0):** one canonical,
  versioned public projection drives bounded managed sections, native
  issue/subissue links and Project 28 fields for every registered consumer
  repository, with dry-run, conflict detection and fail-closed credential
  handling.

### Should have

- **M21 — Finite additive MCDA information value (C17, planned v1.3.0):**
  VOIAGE issue #560 provides exact finite perfect-information value under a
  compensatory additive multi-criteria model with fixed ex-ante value
  functions, declared criterion units and directions, a correlated joint
  outcome/preference law, normalized nonnegative weights, complete ties,
  criterion/preference/joint information actions, gross and signed net value,
  interaction, regret, fractional rank acceptability and Pareto diagnostics.
  A merged experimental Python implementation is repository evidence, not
  scientific approval, stable promotion or Rust/R/Julia parity.
- **M22 — Risk-sensitive and constrained information value (C18, planned
  v1.3.0):** #570 values information through declared risk-sensitive utility or
  risk functionals and feasible policies, with constraints, sampling risk and
  catastrophic outcomes explicit. MoSCoW: Must.
- **M23 — Forecast and signal information value (C18, planned v1.3.0):** #572
  values calibrated forecasts or signals through posterior decision
  consequences rather than accuracy alone. MoSCoW: Should.
- **M24 — Information-source portfolio value (C18, planned v1.3.0):** #582
  selects dependent source portfolios jointly with budgets, compatibility,
  costs, complete ties and solver assurance. MoSCoW: Should.
- **M25 — Information and implementation decomposition (C18, planned
  v1.3.0):** #593 returns current/informed and imperfect/perfect implementation
  cells, EVPIM/EVSIM/EVP/IA-EVSI components and an explicit interaction without
  default independence. MoSCoW: Must.
- **M26 — Uncertainty-modelling value (C18, planned v1.3.0):** #594 values
  resolving model or solution uncertainty separately from acquiring empirical
  information. MoSCoW: Must.
- **M27 — Event-localized information value (C18, planned v1.3.0):** #596
  localizes value to declared events or regions and reconciles to the governed
  aggregate. MoSCoW: Should.
- **M28 — Belief-state sequential information value (C18, planned v1.3.0):**
  #597 values sensing and intervention policies over a declared transition
  model, horizon and stopping rule. MoSCoW: Should.
- **M29 — Signed social and strategic information value (C18, planned
  v1.3.0):** #598 preserves harmful private/social effects instead of clamping
  all information value to nonnegative classical VOI. MoSCoW: Should.
- **M30 — Heterogeneity value decomposition (C18, planned v1.3.0):** #599
  separates known-subgroup policy value from research-on-heterogeneity value,
  with prevalence and interactions explicit. MoSCoW: Should.
- **M31 — Outcome-conditional sample information value (C18, planned
  v1.3.0):** #600 returns realized sample-value distributions and low-value
  risk separately from expected EVSI and estimator error. MoSCoW: Should.
- **M32 — Sampling-acquisition-harm VOI research scope (C18, planned
  v1.3.0):** VOIAGE issue #850 declares each sampling action and explicit
  no-sampling comparator, affected parties, information and acquisition-harm
  laws, timing, units, catastrophic outcomes, mathematical risk criteria and
  constraints. Additive net value is permitted only for a mutually exclusive,
  separable and commensurate outcome ledger; otherwise the result remains
  joint-welfare, constrained or vector. Positive EVSI or ENBS never supplies
  ethics or regulatory authorization. Runtime remains unsupported until an
  exact candidate passes independent scientific/domain review and a separate
  named human verdict. MoSCoW: Must.
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
| v1.2.0 | Must | M16 | C16 / `risk_adjusted_information_pricing_20260731` | VOIAGE #595 and #694–#697, parent #318 |
| v1.2.0 | Must | M17 | C16 / C12 | Project 28 and every registered consumer repository |
| v1.3.0 | Should | M21 | C17 / `supported_frontier_method_completion_20260723` | VOIAGE #560 and #746–#750, parent #318; implementation PR #751 |
| v1.3.0 | Must | M22 | C18 / `supported_frontier_method_completion_20260723` | VOIAGE #570, parent #318 |
| v1.3.0 | Must | M25-M26 | C18 / `supported_frontier_method_completion_20260723` | VOIAGE #593-#594, parent #318 |
| v1.3.0 | Should | M23-M24, M27-M31 | C18 / `supported_frontier_method_completion_20260723` | VOIAGE #572, #582 and #596-#600, parent #318 |
| v1.3.0 | Must | M32 | C18 / `sampling_acquisition_harm_voi_20260802` | VOIAGE #850, child of #570, dependency #571, under #841/#318 |

## Verification

Requirements are evidenced by the track acceptance criteria, deterministic
fixtures, repository harnesses, hosted CI, issue/project reconciliation, and
explicit external-gate records. A green local test never substitutes for a
publication, evidence, credential, reviewer, or governance gate.
