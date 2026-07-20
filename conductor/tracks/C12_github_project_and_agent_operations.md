# C12 — GitHub project, issue automation, and agent operations

**Repositories:** vop_poc_nz, voiage

**Depends on:** C00

## Objective

Maintain a deduplicated issue registry, dependency-aware conductor dashboard, resumable agent state, and safe GitHub project bootstrap.

## Deliverables

- canonical issue backlog
- track registry validator
- status dashboard
- gh CLI project/issue generator
- agent checkpoint protocol

## Acceptance criteria

- [x] Track and issue IDs are unique.
- [x] Dependencies are acyclic and resolvable.
- [x] Agents can resume from local state without replaying completed work.

## Evidence

- C00–C12 issues: https://github.com/edithatogo/vop_poc_nz/issues
- Historical track and development ledgers: issues 26–40
- Shared roadmap: https://github.com/users/edithatogo/projects/28
- MoSCoW requirements: `conductor/requirements.md`
- Mermaid design: `conductor/design.md`

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
