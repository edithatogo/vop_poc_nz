# Supporting design: component attribution of EVoP

A future `voiage` implementation should attribute total EVoP to perspective components such as productivity, patient time, informal care, out-of-pocket costs, whānau spillovers, energy savings, and equity weighting.

The preferred default is a Shapley allocation over component subsets because decision switching makes simple one-at-a-time subtraction order-dependent. For component set \(K\), define the characteristic function as current-information directional EVoP from the base perspective to the base plus subset \(S\). Exact Shapley values are practical for small component sets; permutation sampling can be used for larger sets.

This is an interpretability tool and a response to the “productivity-only” criticism. It remains supporting analysis under conductor track `C02`, not a fourth manuscript headline.
