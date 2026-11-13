# VOP proof-of-concept provider promotion

Date: 2026-11-08. Status: planned; no source migration or scientific acceptance. This supplement maps to the existing C17/C18 method-governance and C19 consumer/assurance tracks; it creates no new lifecycle ID and does not change manifest states. Coordinate with Voiage issue #1286 and PR #1287.

## Boundary

`vop_poc_nz` is an empirical proof of concept. Generic decision, perspective, equity and VOI capability belongs in Voiage after verification. NZ datasets, clinical/institutional assumptions, case studies and research interpretation remain here. Existing method IDs, historical outputs and versioned claims remain traceable; do not relabel a historical prototype result as native Voiage output.

Innovate owns innovation/diffusion/competition/substitution/hype effects. Kairos owns simulation execution. **Paignia** is the working name of the reusable Rust game library incubated in new-drug-reimbursement-game T17; its extraction and package publication remain pending. None is a mandatory new dependency of this POC merely because the boundary is documented. Use a provider only for a justified application need and qualified exact contract.

## Executable leaves

Each leaf changes one behaviour and at most five files. Resolve current canonical package paths before changing the historical `src/` examples. New test names below are proposed, not tests already executed. Write and run the distinguishing failing test before implementation, then focused checks and a commit with evidence.

### [ ] POC01 — method/perspective ownership ledger

Create `conductor/vop-promotion-ledger.json` and `tests/test_vop_promotion_ledger.py`. Inventory actual VOI, perspective/equity and supporting code plus existing Voiage counterparts; classify each as duplicate, reusable-correct, application-specific or unsupported. Preserve C17/C18 method IDs and accepted version boundaries. Test that every generic method has exactly one authoritative provider, that NZ scenarios remain local and that no pending method is labelled migrated. Run `python -m pytest tests/test_vop_promotion_ledger.py` with nonzero tests.

### [ ] POC02 — independent provider reference fixture

For one selected method, add a fixture, independent expected calculation and a provider receipt under the existing test/evidence conventions. A two-strategy equal-probability net-benefit fixture A=[4,0], B=[1,3] has EVPI=1.5; use it only for the corresponding ordinary EVPI contract, not as validation of every specialized method. Test perspective/unit/draw-label mismatch, invalid uncertainty and any method-specific edge conditions. Reject parity with an incorrect prototype as the sole oracle. Voiage implements/fixes the generic provider in its own repository first.

### [ ] POC03 — migrate one consumer adapter

After an accepted Voiage method, modify only the resolved consumer adapter, its focused test, dependency declaration/lock and receipt (split if more than five files). Test exact provider/artifact identity, actual native dispatch, preserved joint draws and unchanged intended case semantics. A missing native backend fails explicitly; do not silently fall back to the POC implementation. Keep legacy reference code reachable only through a clearly labelled research/reference path until its documented retirement, not as a second production library.

### [ ] POC04 — preserve scenarios and closure evidence

Run a frozen NZ scenario through the accepted provider and compare with independent expected quantities and any justified versioned differences. Keep input/evidence provenance, units, population, discounting and perspective explicit. Record source SHAs, package hashes, commands/nonzero counts and unresolved scientific issues. Only then mark that method migrated; preserve C17/C18/C19 and external release/publication gates. Do not delete historical receipts or alter data to force numerical agreement.

## Data and dependency policy

Adopt qualified Rust-provider/Polars/Arrow interfaces for migrated capability, not a blanket claim that this existing Python proof of concept is already native or pandas-free. Any breaking removal of a historical public interface requires its own migration receipt. Reuse existing source repositories rather than duplicate acquisition. No paid compute, private data ingestion, package publication or automatic pin advance is included.

Central policy and identity: repository-standards PR #120 and `docs/paignia-library-boundary.md`. Source ownership and empirical validity remain separate from naming and documentation.
