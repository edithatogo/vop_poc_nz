# Manuscript revision record

| Field | Value |
|---|---|
| Manuscript | `manuscript/jss_submission.tex` |
| Revision round | 1 |
| Date | 2026-07-21 |
| Previous assessment | Major revision recommended |
| Target | Software-methods preprint / journal submission |

| # | Issue | Resolution | Status |
|---:|---|---|---|
| 1 | Synthetic cases could be read as policy evidence | Abstract, case-study methods, results, discussion, ethics, and data statements now label them as synthetic software demonstrations | RESOLVED |
| 2 | VoP treated the societal perspective as implicitly correct | VoP is now directional, indexed by decision and reference perspectives, with a directional loss matrix recommended when no single reference is accepted | RESOLVED |
| 3 | Simplified DCEA equation omitted population shares and full welfare framing | Added Atkinson EDE health, population shares, normalized weights, and a stated boundary between full DCEA and the weighted-NMB approximation | RESOLVED |
| 4 | PSA point estimates lacked precision reporting | Replaced legacy values with a seeded 10,000-draw regeneration, equal-tailed 95% simulation intervals, Wilson decision-probability intervals, and recorded MCSEs | RESOLVED |
| 5 | Reproducibility section cited a nonexistent notebook and stale Conda environment | Replaced with uv/Pixi commands, exact-revision and manifest requirements, and current Python support | RESOLVED |
| 6 | Abstract lacked quantitative findings and scope limits | Added manifest-generated five-model results with uncertainty intervals and a non-policy caveat | RESOLVED |
| 7 | Software novelty claims were absolute | Executed a dated systematic software review with exact queries, deduplication, screening flow, versions, evidence links, exclusions, and a bounded finding | RESOLVED |
| 8 | Māori governance implications were absent | Added te Tiriti and Māori data-sovereignty boundaries, with explicit disclosure that no Māori governance review occurred | RESOLVED |
| 9 | Declarations were incomplete | Added ethics, CRediT-style author contributions, AI-use disclosure, and clarified data availability | RESOLVED |
| 10 | LaTeX emitted a `thumbpdf` warning and produced a 20.7 MB PDF | Removed `thumbpdf`, removed the duplicated raster architecture diagram, and generated non-destructive optimized figure copies | RESOLVED |
| 11 | Inflated adjectives and the "societal bonus" label overstated findings | Replaced with descriptive wording and directional perspective-difference terminology | RESOLVED |
| 12 | Repository contained a 404 page in place of `jss.cls` | Installed the official JSS 3.6 class, bibliography style, logo, and manual; added a journal wrapper; and made the build verify both arXiv and JSS PDFs | RESOLVED |
| 13 | Legacy VoP values were not reproducible from the current directional-loss definition | Recalculated VoP from the declared societal-reference loss for every seeded draw and replaced hand-entered manuscript tables with generated TeX | RESOLVED |
| 14 | Numerical figures were not bound to their inputs | Regenerated the Markov trace, cost-effectiveness planes, NMB violin plot, and CEAC from the same draw set and recorded all hashes in the result manifest | RESOLVED |

Policy-grade empirical validation remains outside this revision. The new
intervals propagate the declared synthetic uncertainty model and must not be
interpreted as empirical uncertainty for the named interventions.
