# Local agent protocol — map first, merge safely, complete with evidence

This protocol applies whenever a coding agent works in local `vop_poc_nz` or `voiage` repositories that may contain unpublished analysis, private inputs, reviewer correspondence, generated outputs, or architecture not visible on GitHub.

## Required sequence

1. Set `VOP_CONDUCTOR_PACK` to the unpacked v6 pack.
2. Run `pack_doctor.py` and `upgrade_plan.py` against the live repository.
3. Run `local_agent_bootstrap.py` to create local maps, audits, prompt series, and track state.
4. Read `conductor_status.md` and select one dependency-ready canonical track.
5. Copy only `safe_add` files automatically. Merge every other item into the live architecture with tests.
6. Promote artifacts explicitly; do not infer that a useful local file belongs on GitHub.
7. Run full native tests plus `run_all_local_gates.py --keep-going`.
8. Mark a track complete only with evidence and commit references.

## Git mutation safety

Mapping and audits are read-only. Before copying or reorganising files, create a named non-default integration branch or worktree and checkpoint tracked changes. `safe_integrate.py` and `local_agent_bootstrap.py --apply-safe` fail closed unless explicit overrides are supplied and recorded. Untracked local research artifacts do not alone block mapping, but they remain subject to publication-boundary classification.

## Public boundary

Public by default:

- package source, tests, schemas, and public documentation;
- synthetic deterministic fixtures;
- non-sensitive case contracts and evidence metadata;
- result manifests containing provenance and hashes rather than protected raw data;
- canonical conductor tracks and issue metadata.

Local or external by default:

- raw or restricted data;
- source PDFs without clear redistribution rights;
- reviewer letters and submission correspondence;
- generated manuscript PDFs/docx and journal staging material;
- profiling products, caches, large result arrays, and exploratory notebooks;
- credentials, tokens, and environment secrets.

## Cross-repository boundary

- `vop_poc_nz` owns the Aotearoa New Zealand empirical/tutorial compendium and preprint reproduction.
- `voiage` owns generalisable production EVoP/PAF contracts and implementations.
- Interoperate through schemas, Arrow/Parquet tables, manifests, CLI/API adapters, and shared fixtures—not hard imports or copied repositories.

## Reorganisation rule

Repository reorganisation is non-destructive, separately committed, and preceded by the hygiene and artifact-promotion plans. Git history replaces ad-hoc backup files; external archives replace large generated/source bundles where appropriate.
