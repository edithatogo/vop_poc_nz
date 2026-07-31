# C19 MoSCoW requirements — planned v1.3.0

## Must

- **C19-M01:** Protect `main`, use least-privilege Actions defaults, require
  immutable approved actions, and preserve a viable solo-maintainer recovery
  path.
- **C19-M02:** Bind governance dispatch and consumer confirmation to an
  immutable SHA/digest/correlation contract with fail-closed receipt handling
  and idempotency.
- **C19-M03:** Make coverage and static-analysis policy truthful through
  non-regressing ratchets and migrate first-party execution to canonical
  package imports.
- **C19-M04:** Make Renovate the sole dependency-update authority after
  validation; gate proposals with lockfile, security, frontier, and test
  evidence.
- **C19-M05:** Remove transient tracked bulk and require an explicit,
  manifest-backed disposition for retained generated artifacts.
- **C19-M06:** Validate governance documentation, track registry, issue
  backlog, design, hosted controls, and Project fields against canonical
  sources.

## Should

- **C19-S01:** Generate deterministic drift receipts and roadmap projections
  from typed canonical records rather than maintaining duplicate prose.
- **C19-S02:** Expand static checks through touched-file or package-wide
  ratchets while keeping fast PR feedback.

## Could

- **C19-C01:** Move suitable large publishable assets to Git LFS or a release
  store after an explicit cost, retention, and provenance decision.
- **C19-C02:** Replace the cross-repository personal token with a narrowly
  scoped GitHub App when its operational cost is justified.

## Won't

- **C19-W01:** C19 will not require a second human reviewer in this
  single-maintainer repository.
- **C19-W02:** C19 will not rewrite history, auto-merge changes, release,
  publish, close issues, or accept risk without separate authorization.
