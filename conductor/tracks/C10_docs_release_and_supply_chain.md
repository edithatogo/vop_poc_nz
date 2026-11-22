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

## C19 closure refinement

Dependency-update authority is planned in
[#61](https://github.com/edithatogo/vop_poc_nz/issues/61): validate Renovate,
then remove Dependabot update automation so one authority governs lockfile,
security, frontier, and hosted-test evidence. Release and publication remain
separate gates.

## Bounded CI repair — 22 November 2026

Assessed PR #93 head `c737df78fc085c6bcf6496c38ad877e3f180f2b2` in an
isolated checkout. This repair keeps the frozen contract bundle and its
migration/staging hashes unchanged:

- The locked PyArrow 25.0.0 writer reproduces bundle 1.0.0 exactly. The failing
  tox environments selected 25.0.1, which changes one Parquet footer byte from
  writer version `25.0.0` to `25.0.1`; decoded tables and schema metadata match.
  Pin the canonical writer in tox, preserving the byte-equality assertion.
  Newer-writer qualification remains a separate dependency-frontier result.
- Require Twine 7 and update its lock entry. Its
  [metadata 2.5 fix](https://twine.readthedocs.io/en/stable/changelog.html#twine-7-0-0-2026-07-27)
  addresses the distribution-check failure. This also covers the capability
  proposed by dependency PR #73; no publication is performed.
- Update the locked Pillow 12.3.0 to 13.0.0, the fixed version reported for the
  42 release-dependency audit findings. Only Pillow and Twine package versions
  change; no advisory is ignored and no audit threshold changes.

Executed with CPython 3.14.10; the lock was generated and checked with the CI's
uv 0.11.29, retaining lock revision 3:

- `tox -e py314 -- tests/test_contract_bundle.py`: 17 passed.
- Locked `pytest --no-cov tests/test_contract_bundle_staging.py
  tests/test_reproducible_build.py tests/test_visualizations_new.py`: 23 passed.
- `uv lock --check` and `scripts/dependency_frontier.py --lane stable --strict`:
  passed.
- The existing `scripts/reproducible_build.py` produced byte-identical wheel
  and source distributions across two builds; Twine 7 checked both successfully.
- The unchanged `pip-audit -r <exported release requirements> --format=json`
  gate examined 37 packages and reported no known vulnerabilities.

The inherited `pack_doctor.py . --pack-root .` attempt still exits 1 because
the external `overlays/vop_poc_nz` pack is unavailable. Local focused results
do not establish all-platform hosted CI, signed publication, provider adoption
or scientific acceptance; the existing acceptance boxes remain unchanged.
