# C19 implementation plan

## Phase 1 — Hosted repository and synchronization assurance

- [ ] **C19-T1 / AC-01:** Implement and evidence the solo-maintainer-compatible
  main ruleset and least-privilege Actions policy in #57.
- [ ] **C19-T2 / AC-02:** Implement immutable dispatch identity, consumer
  receipts, correlation, concurrency, and idempotency in #58.
- [ ] **C19-T3 / AC-05:** Eliminate governance documentation and registry drift
  with deterministic validation in #59.
- [ ] **C19-T4:** Run automated review and full Conductor validation for Phase
  1; retain hosted changes and consumer confirmation as explicit gates.

## Phase 2 — Maintainability and artifact assurance

- [ ] **C19-T5 / AC-03:** Retire first-party legacy imports and introduce
  transparent static/coverage ratchets in #60.
- [ ] **C19-T6 / AC-04:** Replace Dependabot with validated Renovate authority
  and evidence-gated grouping in #61.
- [ ] **C19-T7 / AC-04:** Remove transient tracked bulk and record disposition
  manifests for retained generated artifacts in #62.
- [ ] **C19-T8:** Run automated review, focused tests, package-wide quality
  gates, and artifact-hygiene validation for Phase 2.

## Phase 3 — Reconciliation and closure evidence

- [ ] **C19-T9 / AC-05 / AC-06:** Reconcile the canonical backlog, native
  subissue hierarchy, Project 28 fields, track registry, and Mermaid design.
- [ ] **C19-T10:** Capture exact-head hosted CI, ruleset, Actions, Renovate,
  consumer-receipt, and artifact-hygiene evidence.
- [ ] **C19-T11:** Run full Conductor validation and independent review; leave
  merge, issue closure, release, publication, and risk acceptance as separate
  owner-governed actions.
