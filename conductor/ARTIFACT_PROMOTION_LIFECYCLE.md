# Artifact promotion lifecycle

This lifecycle is now part of the local-agent bootstrap. It prevents unpublished data,
reviewer correspondence, submission artifacts, generated outputs, and exploratory
results from being pushed to GitHub merely because they are useful locally.

## Lifecycle states

1. `local_scratch` — local-only, generated, private, or cache-like material. Keep
   ignored or remove from the git index.
2. `local_reviewed` — possibly useful for publication but requiring a human
   decision, source check, privacy check, and/or licence check.
3. `manifest_backed` — metadata or derived results that can be public only when
   supported by a case contract, evidence ledger, result manifest, model card, or
   schema.
4. `external_artifact` — large, private, submission, or source materials that
   should live outside GitHub, with only hashes/provenance committed where
   appropriate.
5. `public_fixture` — small synthetic, redistributable, deterministic examples
   suitable for docs/tests/tutorials.
6. `public_source` — package source, tests, schemas, conductor tracks, CI, and
   public documentation.

## Required local-agent sequence

Run:

```bash
python "$VOP_CONDUCTOR_PACK/scripts/local_agent_bootstrap.py" . \
  --pack-root "$VOP_CONDUCTOR_PACK" --update-gitignore
python scripts/artifact_promotion.py .
python scripts/publication_gate.py . --strict
```

The bootstrap writes `.conductor/local/artifact_promotion_plan.md`. Agents must
read that file before reorganising the repository or staging files.

## Promotion rules

- A local output can support the manuscript only after a result manifest records
  script, input manifest, software version, random seed, output hash, and runtime.
- A parameter can support the manuscript only after an evidence ledger records
  source, unit, price year, derivation, distribution, and perspective inclusion.
- A source PDF, reviewer letter, submission PDF/docx, or large generated result
  should not be committed directly unless a separate human decision marks it as
  redistributable and necessary.
- Public fixtures should be small, deterministic, non-sensitive, and either
  synthetic or clearly redistributable.

## Why this exists

The `vop_poc_nz` preprint already spans CEA, DCEA, PSA, VOI, BIA, Markov
modelling, and Value of Perspective. The publication risk is not a shortage of
materials; it is uncontrolled scope and unclear boundaries between evidence,
examples, generated outputs, and manuscript claims.
