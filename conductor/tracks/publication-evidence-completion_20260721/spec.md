# C16 specification: Publication evidence completion

## Overview

Close the two remaining evidence gaps for the revised manuscript: regenerate
every displayed analytical result from a versioned manifest with uncertainty
intervals, and replace the scoped software scan with a reproducible systematic
software review if the qualified priority claim remains.

## Functional requirements

1. A single deterministic command regenerates the manuscript result dataset,
   uncertainty summaries, generated TeX table, and included analytical figures.
2. The result manifest records the source revision, package/environment
   versions, input and script hashes, random seed, draw count, uncertainty
   method, schema, and hashes for every generated artifact.
3. A verifier fails closed on stale inputs, missing outputs, hash drift, schema
   drift, or manuscript numerical claims not represented by generated TeX.
4. The manuscript reports uncertainty intervals alongside every estimated
   analytical result and states the interval interpretation and limitations.
5. A dated systematic software-review protocol records sources, search strings,
   eligibility criteria, duplicate handling, screening decisions, versions,
   feature evidence, and review limitations.
6. Any priority claim is retained only when supported by the screened evidence
   and is explicitly bounded by the protocol and search date.

## Non-functional requirements

- Python 3.12--3.14 compatible; deterministic on the same supported platform.
- Generated public artifacts contain synthetic data only.
- Numerical outputs use typed, machine-readable schemas and canonical JSON.
- Tests cover determinism, interval ordering, manifest verification, and stale
  artifact rejection.

## Acceptance criteria

- A clean regeneration followed by verification succeeds.
- A changed input or generated artifact causes verification to fail.
- No hand-entered case-result values remain in the manuscript source.
- The software-review ledger is machine-readable and the manuscript table is
  generated from it.
- Both preprint and JSS PDFs compile without unresolved references or LaTeX
  errors; the arXiv source package compiles independently.

## Out of scope

- External health-economic validation.
- Maori governance review or claims that the synthetic cases establish Maori
  equity effects.
- External submission, release, or publication.

