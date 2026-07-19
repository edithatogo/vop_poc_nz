# C03 — Evidence provenance, model validation, and case contracts

**Repositories:** vop_poc_nz

**Depends on:** C00

## Objective

Make empirical sourcing, derivations, model structure, internal validity, and external-validity status reviewable for every case.

## Deliverables

- case registry and model cards
- evidence and derivation ledgers
- transition/trace validation reports
- external validation and calibration plan

## Acceptance criteria

- [ ] Every manuscript parameter maps to a source or derivation.
- [ ] Every case is labelled synthetic fixture, empirically parameterised tutorial, validation case, or policy-grade evaluation.
- [ ] Model structure and validation limitations are explicit.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
