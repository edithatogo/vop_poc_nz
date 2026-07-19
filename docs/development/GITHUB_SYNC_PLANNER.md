# Conflict-safe GitHub governance planning

`vop_poc_nz.github_sync_planner` performs a pure three-way comparison between
the last applied managed projection, the desired local projection, and a
caller-supplied remote snapshot. It never calls GitHub or applies a change.

The planner owns only the issue title, state, bounded
`<!-- governance:begin -->` / `<!-- governance:end -->` section, managed
labels, and declared Project fields. Text outside the bounded section and
labels not present in the base managed set are preserved as human-owned data.
Stable record markers must match exactly.

Outcomes are `clean`, `local_only`, `remote_only`, or `conflict`. Only a
`local_only` outcome contains a proposed issue payload. Closing is refused
unless the caller explicitly supplies approval. Plans serialize to canonical,
sorted JSON and declare `network_mutation: false`.

The command wrapper accepts exported base and remote JSON snapshots and writes
only under `.conductor/local`:

```powershell
uv run python scripts/plan_github_governance_sync.py `
  --base .conductor/local/github_base.json `
  --remote .conductor/local/github_remote.json `
  --record-id CON-SHR-0013
```

Neither snapshots nor plans are tracked publication artifacts. Private ledger
evidence is already excluded by the concern projection boundary.
