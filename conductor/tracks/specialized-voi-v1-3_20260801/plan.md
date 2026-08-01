# C17 implementation plan

## Phase 1 — Canonical v1.3.0 contract

- [~] **C17-T1:** Freeze M21 `Should`, v1.3.0 traceability, exact finite
  additive-MCDA semantics, exclusions and C16 predecessor boundary. (AC-01,
  AC-02, AC-04)
- [ ] **C17-T2:** Publish the versioned C17 projection with #318, #560,
  #746–#750, PR #751 and Project 28 field expectations. (AC-03)
- [ ] **C17-T3:** Run canonical JSON, Markdown, Mermaid and Conductor registry
  validation. (AC-01–AC-04)

## Phase 2 — Version-aware synchronization

- [ ] **C17-T4:** Extend projection validation and dispatch planning to select
  an explicit versioned projection without changing C16/v1.2.0 semantics.
  (AC-02, AC-03)
- [ ] **C17-T5:** Add backward-compatibility, fail-closed policy, registration,
  conflict and missing-credential tests for both projections. (AC-02, AC-05)
- [ ] **C17-T6:** Run automated review and focused/full repository validation.
  (AC-05)

## Phase 3 — Hosted and external reconciliation

- [ ] **C17-T7:** Record hosted exact-head validation and any authorized
  bounded synchronization result. (AC-05)
- [ ] **C17-T8:** Reconcile Project 28 without automatically merging, closing
  issues or releasing. (AC-03, AC-06)
- [ ] **C17-T9:** Record scientific, stable-promotion, Rust/R/Julia parity,
  release and issue-closure outcomes only when separately evidenced and
  authorized. (AC-06)
