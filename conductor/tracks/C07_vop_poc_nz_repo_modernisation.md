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

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
