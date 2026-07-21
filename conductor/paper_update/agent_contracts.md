# Paper-update agent contracts

Each agent produces a content-addressed receipt and may not promote private
material or alter claims without a cited evidence record.

| Agent | Input | Required output | Hard gate |
|---|---|---|---|
| `literature` | source PDFs/CSLs | source manifest, citation hashes | provenance complete |
| `methods` | contracts, code, fixtures | claim-to-test matrix | every quantitative claim has a manifest |
| `reproducibility` | release tag, lockfiles | reproduction receipt | exact tag and clean environment |
| `editorial` | manuscript + reviews | response matrix, revised draft | no unresolved high claims |
| `integrator` | all receipts | submission manifest | all hashes and versions agree |

Agents are advisory; only the integrator may emit `submission-ready: true`, and
only when every hard gate is satisfied. Private notes, credentials, reviewer
correspondence, and unlicensed source material remain local-only.
