# Perspective Acceptability Frontier contract v1.1.0

The Perspective Acceptability Frontier has two linked layers.

## Acceptability layer

For each perspective or perspective mixture, report the probability that each strategy is optimal across probabilistic draws. Exact ties default to equal probability splitting; first-index tie breaking is available only as an explicit compatibility mode.

## Expected-value frontier

For a convex mixture of two perspectives,

\[
W_\alpha=(1-\alpha)W_L+\alpha W_R, \quad \alpha\in[0,1],
\]

the expected net benefit of each strategy is linear in \(\alpha\). The expected-value frontier is therefore the upper envelope of these lines. Switch points should be calculated from exact pairwise intersections and envelope segments, not inferred solely from an arbitrary grid.

## Reporting

Report mixture weights, optimal strategy set, probability-optimal curves, exact switch points, ties, expected net benefit, welfare scale, threshold, and contract version. A threshold-by-mixture surface is a supporting extension, not an additional headline concept.

## Endpoint and tie semantics

The exact expected-value frontier is a set of closed mixture intervals separated by analytically calculated switch points. A zero-width tie at a switch point should be reported as a boundary event rather than expanded into a spurious interval. Probability-optimal curves use the same declared tie policy as the associated EVoP analysis.

## Numerical assurance

The exact frontier and a dense grid implementation should agree away from switch points. Cross-package conformance fixtures must cover single switches, no-switch cases, endpoint switches, and exact ties.
