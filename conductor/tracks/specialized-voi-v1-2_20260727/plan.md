# C16 implementation plan

## Phase 1 — Canonical contract

- [x] **C16-T1:** Freeze M14–M17, MoSCoW priorities and v1.2.0 traceability.
- [x] **C16-T2:** Publish the specialized method and synchronization Mermaid
  designs.
- [x] **C16-T3:** Validate canonical and consumer track cross-references.

## Phase 2 — GitHub projection

- [x] **C16-T4:** Reconcile #313 > #318 > #571/#595/#619 without duplicates.
- [x] **C16-T5:** Set Project 28 MoSCoW, Contract Version, Track ID, Record ID,
  evidence and Sync State fields.
- [x] **C16-T6:** Add the canonical versioned projection to the conflict-safe
  cross-repository synchronization input.

## Phase 3 — Consumer automation

- [x] **C16-T7:** Add a fail-closed consumer registration and dispatch path
  that creates bounded projection-mirror synchronization proposals in
  registered repositories; managed issue and Project updates remain separately
  authorized three-way plans. [ba07539]
- [x] **C16-T8:** Test missing credentials and unregistered repositories; retain
  the existing planner's remote-human-edit and three-way-conflict tests for
  managed issue and Project update proposals. [ba07539]
- [~] **C16-T9:** Run repository and hosted validation, retaining merge,
  release and issue closure as separate gates.
