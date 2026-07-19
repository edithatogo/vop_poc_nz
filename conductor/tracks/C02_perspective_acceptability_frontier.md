# C02 — Perspective Acceptability Frontier and interpretability

**Repositories:** voiage, vop_poc_nz

**Depends on:** C01

## Objective

Productionise a tie-aware PAF with exact expected-value frontier segments, mixture-weight switch points, threshold-surface planning, and component attribution as a supporting analysis.

## Deliverables

- tie-aware acceptability probabilities
- exact mixture frontier segments
- switch-point API and plots
- scoped Shapley component-decomposition design

## Acceptance criteria

- [ ] Switch points are not grid-resolution artefacts.
- [ ] Ties are handled explicitly rather than silently by array order.
- [ ] Frontier outputs include method-contract version and decision metadata.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
