# Monte Carlo assurance for EVoP and PAF

Monte Carlo assurance is supporting infrastructure rather than a new manuscript concept.

## Required outputs

For each principal EVoP estimate, retain:

- the point estimate and decision rule;
- the number of PSA draws;
- bootstrap standard error and confidence interval, or an equivalent simulation-error analysis;
- a convergence profile across increasing draw counts;
- the random seed and software/method-contract versions;
- the frequency and handling of ties;
- the probability of perspective-driven decision discordance.

## Interpretation boundary

A narrow Monte Carlo interval only shows that the numerical estimator is stable for the supplied model and distributions. It does not show that the model is externally valid or that a perspective is normatively correct.

## Promotion rule

Diagnostics may remain local during development. Before a quantitative claim is promoted into the preprint, journal manuscript, or public tutorial, its summary and result-manifest hash should be public or externally archived.
