# EVoP v6 integration notes

The reference implementation corrects the decision object: strategies are alternatives and perspectives are evaluative lenses.

## Required live-pipeline changes

- construct a draw × strategy × perspective net-benefit tensor;
- report current-information directional EVoP as the primary quantity;
- label per-draw perspective regret as a diagnostic;
- declare tie policy and method-contract version;
- add exact PAF switch points and tie-aware probability-optimal curves;
- attach bootstrap/convergence diagnostics to principal reported EVoP values;
- store public outputs in manifest-backed Arrow/Parquet/JSON rather than pickle;
- update manuscript language so delta NMB or a “societal bonus” is not called EVoP.

The local agent must merge these changes into the existing pipeline after mapping the live worktree. This file does not assert that the public/local pipeline has already been updated.
