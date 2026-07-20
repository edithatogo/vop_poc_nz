# C05 — Pipeline reproducibility, manifests, and artifact governance

**Repositories:** vop_poc_nz, voiage

**Depends on:** C00, C01

## Objective

Ensure every result, figure, table, and release artifact has traceable inputs, code, seeds, environment, hashes, and promotion status.

## Deliverables

- no-orphan-result manifests
- Parquet/Arrow/JSON interchange
- local-to-public artifact lifecycle
- release snapshot and run ledger
- simulation assurance artifacts for EVoP/PAF estimates

## Acceptance criteria

- [ ] No manuscript output lacks a manifest.
- [ ] Pickle is not the public interchange format.
- [ ] Private/local artifacts cannot pass the publication gate by default.
- [ ] Monte Carlo diagnostics record draws, seeds, method version, and manifest hashes.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
