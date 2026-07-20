# Project workflow

## Sources of truth

1. `conductor/manifest.json` defines track identity and dependencies.
2. Each active track's `spec.md` defines acceptance criteria.
3. Each active track's `plan.md` records test-first execution state.
4. `.conductor/local/track_state.json` records local evidence and commits.
5. `conductor/LOCAL_AGENT_PROTOCOL.md` defines map-first and publication safety.

## Task lifecycle

For every task: mark it `[~]`, add a failing test or contract check, implement
the smallest compatible change, run focused checks, refactor, run the relevant
full gates, commit the functional change, record its SHA/evidence, then mark it
`[x]`. Never mark external approval or hardware evidence complete locally.

## Required quality gates

- Ruff, ty and BasedPyright pass under their documented debt policies.
- Unit, property, integration, E2E and contract tests pass.
- New critical code has at least 95% coverage; repository coverage does not
  regress.
- Arrow fixtures and cross-repository compatibility hashes remain valid.
- Dependency, security, build, release and repository harness checks pass.
- Performance-sensitive changes satisfy benchmark budgets and scheduled
  Scalene evidence is reproducible.
- Public API migrations include compatibility adapters, warnings and tests.

