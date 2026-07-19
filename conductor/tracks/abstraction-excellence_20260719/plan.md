# Implementation plan: Domain Abstraction Excellence

## Phase 1 - Governance and executable contracts

- [x] Task: Add failing tests for typed concern and core domain schemas. `51a6134`, `2566c0c`
- [x] Task: Implement concern, parameter, numerical policy, run context and result models. `51a6134`, `2566c0c`
- [x] Task: Export deterministic JSON Schemas and shared contract metadata. `51a6134`
- [x] Task: Phase verification and checkpoint. C13 hardening contracts and focused harness verified 2026-07-19. `7faddc8`

## Phase 2 - Calculation and backend adoption

- [x] Task: Add failing protocol and capability-parity tests. `2566c0c`, VOIAGE `08fc460`
- [x] Task: Implement generic calculation-kernel and backend capability contracts. `2566c0c`, VOIAGE `08fc460`
- [x] Task: Adapt VOP and VOIAGE calculations without API breakage. `bb3bda9`, VOIAGE `835dccf`
- [x] Task: Phase verification and checkpoint. Typed/legacy numerical parity and pure-kernel regressions verified 2026-07-19. `7faddc8`

## Phase 3 - VOP canonical package migration

- [x] Task: Inventory duplicate modules and import consumers. `09812ee`
- [x] Task: Add migration and deprecation-shim tests. `09812ee`, `02982e5`
- [x] Task: Consolidate canonical implementations under `vop_poc_nz`. `09812ee`, `02982e5`
- [x] Task: Separate pipeline calculation, orchestration, reporting and I/O boundaries. `bb3bda9`
- [x] Task: Phase verification and checkpoint. Fresh-process import boundary is side-effect free and verified 2026-07-19. `7faddc8`

## Phase 4 - Concern ledger and GitHub synchronization

- [x] Task: Add privacy-safe concern-ledger fixtures and validation tests. `51a6134`, `d7cd1a9`
- [x] Task: Implement issue/Project synchronization payload generation. `55b3e5b`
- [x] Task: Synchronize C13 and historical relationships into GitHub issues/project. Issue `#41`, Project `#28`
- [x] Task: Phase verification and checkpoint. Human metadata preservation and conflict-safe planning verified 2026-07-19. `7faddc8`

## Phase 5 - Documentation, CI and closeout

- [x] Task: Update MoSCoW requirements and Mermaid design in both repositories. `492c951`, VOIAGE `d8d0d1c`
- [x] Task: Extend CI, profiling, mutation, type and contract gates. `b250360`, VOIAGE `f51d0e1`
- [ ] Task: Run focused and full local gates plus cross-repository conformance.
- [ ] Task: Review compatibility, security, performance and release evidence.
- [ ] Task: Phase verification and checkpoint.

## 2026-07-19 C13 principal-engineer review remediation evidence

- Implementation commit: `7faddc8`.
- `uv run pytest tests/test_c13_contract_hardening.py tests/test_typed_cea_contract.py tests/test_typed_pipeline.py tests/test_github_sync_planner.py tests/test_governance_harness.py tests/test_import_boundary.py -q --no-cov -p no:cacheprovider` - 34 passed.
- `uv run pytest tests/test_core_functions.py tests/test_cea_model_core_coverage.py tests/test_compatibility_contract.py tests/test_perspective_conformance.py -q --no-cov -p no:cacheprovider` - 25 passed.
- `uv run python scripts/governance_harness.py . --strict` - schema/ledger, focused tests, import boundary, Ruff, Ruff format, BasedPyright and ty all passed.
- Full-repository, cross-repository, hosted CI, security, release and figure-heavy evidence remain open Phase 5 gates and are not claimed by this checkpoint.
