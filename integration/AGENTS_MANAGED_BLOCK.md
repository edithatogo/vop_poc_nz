<!-- BEGIN VOP-CONDUCTOR MANAGED BLOCK -->
## Conductor map-first workflow

Before editing this repository:

1. Set the unpacked implementation pack path, for example `export VOP_CONDUCTOR_PACK=/path/to/vop_voiage_conductor_implementation_v6`.
2. Run `python "$VOP_CONDUCTOR_PACK/scripts/pack_doctor.py" . --pack-root "$VOP_CONDUCTOR_PACK"` and read `.conductor/local/pack_doctor.md`.
3. Run `python "$VOP_CONDUCTOR_PACK/scripts/local_agent_bootstrap.py" . --pack-root "$VOP_CONDUCTOR_PACK" --update-gitignore`.
4. Inspect `upgrade_plan.md`, `repo_map.md`, `metadata_consistency.md`, and `repo_hygiene.md` before changing files.
5. Create a dedicated integration branch or worktree before any mutating integration. The safety guard refuses default branches, detached HEAD, and tracked/staged changes unless an override is explicit and recorded.
6. Work only on a dependency-ready canonical track from `conductor/manifest.json`; record status, evidence, and commits in `.conductor/local/track_state.json`.
7. Never overwrite an existing implementation merely because an overlay contains a file with the same purpose. Merge reference contracts and fixtures into the native architecture.
8. Keep raw data, reviewer correspondence, generated outputs, submission files, credentials, and exploratory artifacts local unless explicitly promoted through the artifact ledger.
9. Before commit, release, arXiv update, or journal submission, run `python scripts/run_all_local_gates.py . --pack-root "$VOP_CONDUCTOR_PACK" --keep-going` and resolve or explicitly document every failure.

For `vop_poc_nz`, keep generalisable EVoP/PAF methods compatible with the canonical `voiage` contract. For `voiage`, integrate into the existing API, registry, CLI, fixtures, and binding conventions rather than creating a parallel perspective package.
<!-- END VOP-CONDUCTOR MANAGED BLOCK -->
