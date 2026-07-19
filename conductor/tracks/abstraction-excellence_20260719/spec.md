# Specification: Domain Abstraction Excellence

## Overview

Unify the VOP-VOIAGE domain architecture so concerns, parameters,
calculations, backends and results are explicit, immutable, machine-readable,
portable and provenance bearing without breaking supported public APIs.

## Functional requirements

1. Define frozen Pydantic v2 models for concern, assumption, risk, decision,
   evidence and issue links, with discriminated types and generated JSON Schema.
2. Define typed parameter, distribution, analysis, numerical-policy and run
   context specifications with units, dimensions, provenance and validation.
3. Define generic calculation-kernel and backend-capability protocols.
4. Define a generic analysis-result envelope carrying diagnostics, maturity,
   contract/version identity, provenance and Arrow schema identity.
5. Adopt the contracts in VOP perspective and pipeline boundaries and VOIAGE
   perspective/backend boundaries through compatibility adapters.
6. Make `vop_poc_nz` the canonical VOP import tree and replace duplicate
   top-level implementations with tested deprecation shims or remove them when
   no supported consumer remains.
7. Generate/synchronize GitHub issue and Project metadata from the typed
   concern ledger without publishing local/private evidence.

## Non-functional requirements

- Python 3.14; Pydantic v2; Arrow/Polars/xarray/NumPy interoperability.
- Stable APIs remain compatible for the current major version.
- Experimental features are opt-in, capability checked and maturity governed.
- Deterministic JSON Schema, Arrow fingerprints and fixtures work in a fresh
  process and across both repositories.
- Structured logging contains run/analysis/contract identifiers with redaction.
- CI covers lint, formatting, ty, BasedPyright, unit, property, integration,
  E2E, contract, mutation, security, build, release and profiling evidence.

## Acceptance criteria

- No untyped dictionary fields or forwarding `**kwargs` remain in the new
  canonical public contracts.
- Calculation code is separated from orchestration, reporting and I/O.
- Every legacy path has a tested migration or explicit removal decision.
- Requirements retain MoSCoW prioritization and design documents contain
  current Mermaid component and execution flows.
- Both focused and full repository gates pass and hosted CI evidence is green.

## Out of scope

- Publishing private/local evidence.
- Claiming stable accelerator support without parity and hardware evidence.
- Removing a documented public API without a compatibility window.

