# vop_poc_nz product definition

## Vision

Provide an Aotearoa New Zealand demonstration, research and tutorial compendium
for value of perspective and value of heterogeneity in health-economic
decisions. Develop application cases and contribute reusable method contracts,
independent fixtures and improvements to VOIAGE for operational use.

## Ownership boundary

- VOP owns case studies, evidence metadata, application pipelines, perspective
  and subgroup assumptions, policy interpretation, and the application side of
  versioned VOP-VOIAGE compatibility cases.
- VOIAGE owns reusable value-of-perspective, value-of-heterogeneity and
  information-value method contracts, calculation APIs, backend implementations,
  registries and cross-language bindings. Existing application/reference
  calculations are migrated only with accepted provider and consumer evidence.
- Paignia owns generic game representations and supported solvers when a case
  requires strategic interaction; UOGTO owns canonical game semantics. Neither
  ordinary perspective comparison nor subgroup analysis requires a game runtime.
- Repositories interoperate through versioned schemas, Arrow artifacts,
  deterministic fixtures and optional public adapters. Providers do not import
  this application's source or domain assumptions.

The existing [C09 plan](tracks/C09_cross_repo_boundaries_and_adapters.md) records
successive Voiage adoption and any game-provider composition. This scope
clarification does not complete that plan or promote a method's maturity.

## Excellence criteria

Public analysis inputs and outputs are typed, immutable, validated, provenance
bearing, backend explicit, numerically reproducible, and migration compatible.
Concerns, assumptions, risks, decisions, evidence and issue links are
machine-readable. Experimental features are capability-gated and never imply
unsupported maturity.

