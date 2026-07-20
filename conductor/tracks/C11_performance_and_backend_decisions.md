# C11 — Performance and backend decisions

**Repositories:** vop_poc_nz, voiage

**Depends on:** C01, C05

## Objective

Profile before optimising and adopt Polars/Arrow, JAX/XLA/NumPyro, Mojo, or Rust only where measured evidence and compatibility justify them.

## Deliverables

- Scalene baseline
- benchmark and numerical-equivalence suite
- backend ADRs
- optional accelerator prototypes

## Acceptance criteria

- [ ] Correctness and conformance precede optimisation.
- [ ] Every backend change has benchmark and equivalence evidence.
- [ ] Experimental backends do not expand the manuscript scope.

## 2026-07-19 implementation evidence

- `scripts/profile_workload.py` is a deterministic Arrow/Polars workload.
- `pixi run profile` and the scheduled/manual CI frontier produce Scalene
  artifacts from that workload.
- Free-threaded and experimental dependency lanes remain observational and
  opt-in; stable CI retains correctness, interchange, and numerical budgets.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
