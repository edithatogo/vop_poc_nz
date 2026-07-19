# Implementation plan: Domain Abstraction Excellence

## Phase 1 - Governance and executable contracts

- [x] Task: Add failing tests for typed concern and core domain schemas. `51a6134`, `2566c0c`
- [x] Task: Implement concern, parameter, numerical policy, run context and result models. `51a6134`, `2566c0c`
- [x] Task: Export deterministic JSON Schemas and shared contract metadata. `51a6134`
- [ ] Task: Phase verification and checkpoint.

## Phase 2 - Calculation and backend adoption

- [x] Task: Add failing protocol and capability-parity tests. `2566c0c`, VOIAGE `08fc460`
- [x] Task: Implement generic calculation-kernel and backend capability contracts. `2566c0c`, VOIAGE `08fc460`
- [x] Task: Adapt VOP and VOIAGE calculations without API breakage. `bb3bda9`, VOIAGE `835dccf`
- [ ] Task: Phase verification and checkpoint.

## Phase 3 - VOP canonical package migration

- [x] Task: Inventory duplicate modules and import consumers. `09812ee`
- [x] Task: Add migration and deprecation-shim tests. `09812ee`, `02982e5`
- [x] Task: Consolidate canonical implementations under `vop_poc_nz`. `09812ee`, `02982e5`
- [x] Task: Separate pipeline calculation, orchestration, reporting and I/O boundaries. `bb3bda9`
- [ ] Task: Phase verification and checkpoint.

## Phase 4 - Concern ledger and GitHub synchronization

- [x] Task: Add privacy-safe concern-ledger fixtures and validation tests. `51a6134`, `d7cd1a9`
- [x] Task: Implement issue/Project synchronization payload generation. `55b3e5b`
- [x] Task: Synchronize C13 and historical relationships into GitHub issues/project. Issue `#41`, Project `#28`
- [ ] Task: Phase verification and checkpoint.

## Phase 5 - Documentation, CI and closeout

- [x] Task: Update MoSCoW requirements and Mermaid design in both repositories. `492c951`, VOIAGE `d8d0d1c`
- [x] Task: Extend CI, profiling, mutation, type and contract gates. `b250360`, VOIAGE `f51d0e1`
- [ ] Task: Run focused and full local gates plus cross-repository conformance.
- [ ] Task: Review compatibility, security, performance and release evidence.
- [ ] Task: Phase verification and checkpoint.
