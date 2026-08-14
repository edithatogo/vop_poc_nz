# C07 — vop_poc_nz repository modernisation and cleanup

**Repositories:** vop_poc_nz

**Depends on:** C00

## Objective

Turn the proof-of-concept into a clean empirical compendium and tutorial package while preserving local research artifacts outside the public package boundary.

## Deliverables

- root-cleanup migration plan
- Python 3.14/Pixi/tooling decision
- package import/API cleanup
- README/homepage/release metadata repair

## Acceptance criteria

- [ ] Generated logs, archives, site builds, and private submission assets are not tracked at repo root.
- [ ] README examples import the installed package.
- [ ] Version, licence, citation, Python support, and release metadata agree.

## 2026-07-19 implementation evidence

- `pixi.toml` delegates reproducible cross-platform tasks to the canonical
  `uv.lock` environment.
- Hatch VCS derives build and runtime versions from reviewed Git tags.
- Pydantic v2 logging settings provide JSONL, run IDs, bound context, and
  non-destructive handler ownership.
- `.github/workflows/quality-frontier.yml` makes Ruff, BasedPyright, `ty`,
  focused tests, package smoke tests, Scalene, mutation, audit, and experimental
  evidence visible.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.

## C19 closure refinement

C19 turns the remaining C07 gaps into explicit planned-v1.3.0 work:

- canonical import retirement and transparent quality ratchets: [#60](https://github.com/edithatogo/vop_poc_nz/issues/60);
- generated/debug artifact disposition and prevention: [#62](https://github.com/edithatogo/vop_poc_nz/issues/62).

These tasks are nested under #55 > #53 and do not invalidate the 2026-07-19
implementation evidence above.
