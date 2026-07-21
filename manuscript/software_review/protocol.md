# Systematic software-review protocol

The review asks whether reusable health-economic evaluation software exposes a
named directional Value of Perspective (VoP) workflow and maps adjacent native
multi-perspective and DCEA capabilities. Searches were executed on 21 July 2026
through the GitHub repository search API with the exact strings recorded in
`systematic-software-review.json`. The first 50 best-match results per query
were exported, deduplicated by repository name, and screened against the
pre-specified eligibility criteria.

Title/description screening removed tutorials, course repositories, individual
applied models, unrelated repositories, and curated lists. Full-text screening
used repository README files, tagged documentation, package websites, and
release metadata. “No” means no named feature was located in inspected public
documentation; it does not mean the calculation is impossible using a package.
“Manual” means analysts can represent perspectives through separate inputs or
runs but no dedicated simultaneous comparison was located. The review was not
independently duplicated, and GitHub ranking/search limits can omit software.

All flow counts, decisions, versions, evidence URLs, and limitations are stored
in the machine-readable ledger. The manuscript table is generated from that
ledger and must not be edited by hand.

