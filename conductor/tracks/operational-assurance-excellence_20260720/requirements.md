# C15 requirements: Operational Assurance Excellence

## MoSCoW

### Must

- **C15-M01:** Trusted governance-baseline capture shall require explicit reviewer
  approval and emit immutable source-revision evidence.
- **C15-M02:** Standalone bundle staging shall bind exact content, tag, SBOM, and
  provenance while remaining private until authorized publication.
- **C15-M03:** Current, N-1, and incompatible migration cases shall be tested by
  the independent consumer without importing producer runtime code.
- **C15-M04:** Aggregate, critical-module, and changed-line branch coverage shall
  be enforced without exclusions that conceal new production behaviour.
- **C15-M05:** Mutation evidence shall identify its source cohort and retain score,
  absolute unresolved debt, density, universe changes, and promotion provenance.
- **C15-M06:** Reproducibility shall compare normalized artifact digests from
  independent Linux and Windows builders.
- **C15-M07:** Performance evidence shall include runner identity, repeated samples,
  confidence intervals, and deterministic regression decisions.
- **C15-M08:** Collector-level telemetry tests shall prove correlation and recursive
  secret redaction in exported payloads.
- **C15-M09:** Independent scientific oracles shall cover boundary, near-tie,
  extreme-tail, and higher-dimensional cases with units and tolerances.
- **C15-M10:** Exact-head local and hosted assurance plus independent review shall
  pass before repository-owned completion.

### Should

- **C15-S01:** Bundle staging should use OCI-compatible descriptors when available.
- **C15-S02:** Differential coverage should annotate pull requests without write
  permission and retain a machine-readable artifact.
- **C15-S03:** Performance comparisons should use bootstrap confidence intervals
  where the sample size supports them.

### Could

- **C15-C01:** A transparency-log receipt may be attached after approved signing.
- **C15-C02:** Hardware-specific performance cohorts may be added when trustworthy
  runners are available.

### Won't

- **C15-W01:** C15 will not autonomously trust governance state, merge, publish,
  sign, close issues, or mutate Project truth beyond explicitly requested sync.
- **C15-W02:** Experimental dependencies will not replace the frozen stable lane.
