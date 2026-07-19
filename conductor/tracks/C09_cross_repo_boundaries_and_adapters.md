# C09 — Cross-repository boundaries and adapters

**Repositories:** vop_poc_nz, voiage

**Depends on:** C00, C05, C08

## Objective

Define typed interchange with related repositories without hard coupling or copying whole codebases.

## Deliverables

- import-boundary gate
- Arrow/Parquet/JSON schemas
- adapter decision records
- conformance fixtures for selected integrations

## Acceptance criteria

- [ ] vop_poc_nz does not import voiage internals unless explicitly declared.
- [ ] Related repos communicate through stable contracts.
- [ ] Adapters are optional and independently testable.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
