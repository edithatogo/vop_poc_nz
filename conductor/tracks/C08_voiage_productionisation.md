# C08 — voiage productionisation of perspective methods

**Repositories:** voiage

**Depends on:** C01, C02

## Objective

Integrate EVoP and PAF into the existing voiage architecture and frontier contracts rather than adding a parallel package surface.

## Deliverables

- native voiage API/CLI integration
- registry-backed fixtures
- MCDA feature export boundary
- binding and serialization contracts

## Acceptance criteria

- [ ] No duplicate perspective implementation is introduced.
- [ ] Existing voiage schemas/CLI/registry conventions are followed.
- [ ] Cross-language fixtures remain deterministic.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
