# Mutation baseline promotion

The C15 mutation cohort cannot approve itself. CI compares the exact bytes of
`vop-c15-cohort.json` with the protected repository variable
`VOP_MUTATION_BASELINE_SHA256`.

Promotion requires a maintainer to download the hosted universe and cohort
evidence, review every added/removed mutant plus score, absolute debt, debt
density, source/configuration hashes, locked Mutmut version, and lock hash, and
then update the candidate through normal reviewed repository governance. Do not
add an approval Boolean to the baseline. Only after the reviewed commit is
authoritative should the SHA-256 of the exact baseline bytes be installed as the
protected repository variable. Until the captured universe and external digest
both exist, the gate retains candidate evidence and fails closed.
