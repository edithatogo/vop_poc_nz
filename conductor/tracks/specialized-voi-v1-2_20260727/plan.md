# C16 implementation plan

## Phase 1 — Canonical contract

- [ ] **C16-T1:** Freeze M14–M17, MoSCoW priorities and v1.2.0 traceability.
- [ ] **C16-T2:** Publish the specialized method and synchronization Mermaid
  designs.
- [ ] **C16-T3:** Validate canonical and consumer track cross-references.

## Phase 2 — GitHub projection

- [ ] **C16-T4:** Reconcile #313 > #318 > #571/#595/#619 without duplicates.
- [ ] **C16-T5:** Set Project 28 MoSCoW, Contract Version, Track ID, Record ID,
  evidence and Sync State fields.
- [ ] **C16-T6:** Add the canonical versioned projection to the conflict-safe
  cross-repository synchronization input.

## Phase 3 — Consumer automation

- [ ] **C16-T7:** Add a fail-closed consumer registration and dispatch path
  that creates bounded managed-section synchronization proposals in registered
  repositories.
- [ ] **C16-T8:** Test missing credentials, unregistered repositories, remote
  human edits and three-way conflicts.
- [ ] **C16-T9:** Run repository and hosted validation, retaining merge,
  release and issue closure as separate gates.
