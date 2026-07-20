# C10 — Documentation, release truth, and software supply chain

**Repositories:** vop_poc_nz, voiage

**Depends on:** C05, C07, C08

## Objective

Create a coherent public surface and verifiable releases across PyPI, conda, documentation, citation, archival, and provenance services.

## Deliverables

- metadata consistency gate
- Astro/Starlight documentation decision
- PyPI/conda publication workflow
- SBOM, attestations, trusted publishing, archive reconciliation

## Acceptance criteria

- [ ] README, pyproject, CITATION, licence, tags, and package version agree.
- [ ] Release artifacts are reproducible and signed/attested where supported.
- [ ] Documentation clearly separates stable, experimental, and local-only features.

## Agent operating rule

Map the live worktree and inspect existing implementations before editing. Prefer the smallest compatible change, record evidence in the local track state, and never promote local/private artifacts implicitly.
