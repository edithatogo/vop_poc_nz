# Directional EVoP method contract v1.1.0

## Decision object

Strategies are the alternatives. Perspectives are evaluative lenses. For parameter draw \(\theta\), strategy \(d\), and perspective \(p\), let welfare-scaled net benefit be \(W_p(d,\theta)\).

## Current-information EVoP

The policy decision under perspective \(p\) is fixed using expected welfare:

\[
d_p = \arg\max_d E_\theta[W_p(d,\theta)].
\]

Directional Expected Value of Perspective from \(p\) to \(q\) is:

\[
EVoP(p\rightarrow q)=E_\theta[W_q(d_q,\theta)-W_q(d_p,\theta)].
\]

This is the primary decision parameter. It is directional, non-negative up to numerical tolerance, and zero when the fixed decisions coincide.

## Per-draw diagnostic

A separate diagnostic allows the optimal strategy to vary by draw:

\[
DVoP(p\rightarrow q)=E_\theta[W_q(d_q(\theta),\theta)-W_q(d_p(\theta),\theta)].
\]

This quantity mixes parameter information with perspective discordance and must not be silently labelled as current-information EVoP.

## Required metadata

Every reported value must include source perspective, evaluation perspective, decision rule, strategy set, willingness-to-pay or welfare scale, population scaling, price year/currency where monetary, tie policy, method-contract version, and a result-manifest reference.

## Invariants

- `EVoP(p -> p) = 0`.
- Values are directional and need not be symmetric.
- Population EVoP equals per-person EVoP times the declared population multiplier.
- Perspectives are never treated as decision alternatives.
- Exact ties are handled explicitly and reported.

## Tie policies

A tie is part of the decision rule, not a harmless software detail.

- `split`: the default analytical policy for reporting; tied strategies receive equal selection weight.
- `first`: a deterministic compatibility policy that must be labelled because results can depend on strategy ordering.
- `error`: fail closed so a decision-maker must resolve the tie explicitly.

For current-information EVoP, a split source-perspective tie averages target-perspective welfare over the tied source-optimal strategies. A target-perspective tie averages over target-optimal strategies; their expected target welfare is equal, although their draw-level welfare may differ.

## Monte Carlo estimator uncertainty

The PSA estimate of EVoP has simulation error. Reports intended for publication should include at least one of:

- a non-parametric bootstrap standard error and percentile interval that re-runs the decision rule after resampling draws; or
- a documented convergence profile over increasing draw counts.

These quantify finite-simulation uncertainty only. They do not resolve uncertainty about evidence, model structure, welfare scale, or which perspective should be authoritative.
