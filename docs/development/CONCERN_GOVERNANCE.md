# Concern governance

`vop_poc_nz.concerns` defines the privacy-safe governance boundary for C13.
The strict, frozen Pydantic models distinguish unresolved concerns, relied-on
assumptions, uncertain risks, approved decisions, provenance-bearing evidence,
and external coordination links.

The canonical ledger remains local data. `build_github_sync_payloads()` only
constructs deterministic desired-state payloads: it does not import a GitHub
client, read credentials, call a network service, or mutate an issue or
Project. Records and evidence marked `local_private` are excluded from those
payloads. Applying a payload requires a separate explicitly authorised process.

Generate the published JSON Schemas with:

```powershell
uv run python scripts/generate_concern_governance_schemas.py
```

Schema output is sorted, UTF-8, and newline-normalised so regeneration is
reviewable. A GitHub issue is a coordination projection, not the source of
truth and not evidence by itself.
