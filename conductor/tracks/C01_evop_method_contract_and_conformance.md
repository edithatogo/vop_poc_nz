# C01 — Directional EVoP method contract and cross-package conformance

**Repositories:** vop_poc_nz, voiage

**Depends on:** C00

## Objective

Lock the mathematical and software contract for current-information EVoP, per-draw diagnostics, decision discordance, scaling, tie behaviour, and output metadata.

## Deliverables

- versioned method specification
- shared deterministic fixtures
- cross-package parity tests
- migration away from perspective-as-alternative calculations
- Monte Carlo standard errors, bootstrap intervals, and draw-count convergence diagnostics

## Acceptance criteria

- [ ] Current-information and per-draw quantities have different names and tests.
- [ ] EVoP is directional and zero under identical source/target perspectives.
- [ ] Both repositories pass the same conformance fixtures.
- [ ] Tied decisions have declared `split`, `first`, or `error` semantics.
- [ ] Principal EVoP estimates have simulation-error or convergence evidence.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
