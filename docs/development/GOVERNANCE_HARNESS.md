# Governance automation harness

C13 governance checks have one bounded command surface:

```powershell
uv run python scripts/governance_harness.py . --strict
```

The harness verifies deterministic schema regeneration, validates the canonical
ledger, runs only the concern, sync-planner, harness, and import-boundary tests,
checks the contract-first import boundary, and applies Ruff, formatting, and
BasedPyright to the new governance modules and scripts. Its report is written
under `.conductor/local` and is never uploaded by CI.

The same command is used by the existing Quality Frontier workflow, the Pixi
`governance` task, and the broader local-gate runner. This avoids parallel CI
definitions and keeps uv as the single Python resolver beneath Pixi.

Profiling and mutation remain scheduled or manually dispatched evidence lanes.
The governance profile is capped at 1,000 CI iterations and records aggregate
timing, outcome, and digest data only. Mutation is limited to the logging and
sync-planner modules with their focused tests. Neither lane contains a GitHub
client, publishes `.conductor/local`, or promotes private ledger content.
