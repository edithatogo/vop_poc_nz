# C00 — Repository intake, mapping, and agent bootstrap

**Repositories:** vop_poc_nz, voiage

**Depends on:** None

## Objective

Map the actual local worktrees before any patching, classify local-only artifacts, detect existing implementations, and initialise resumable conductor state.

## Deliverables

- repo map and pack-doctor report
- managed AGENTS.md conductor block
- track state initialisation
- safe integration plan with no overwrites
- dedicated branch/worktree and Git mutation-safety report

## Acceptance criteria

- [ ] No source file is changed before repo mapping completes.
- [ ] Existing perspective/conductor implementations are detected and marked merge-required.
- [ ] Local/private/generated artifacts are classified before Git staging.
- [ ] Mutating integration occurs on a named non-default branch/worktree with a clean tracked index, or an explicit override is recorded.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
