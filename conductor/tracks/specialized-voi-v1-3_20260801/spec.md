# Finite additive MCDA information-value governance

## Overview

C17 is the canonical v1.3.0 continuation of completed C16. It governs M21 and
VOIAGE issue #560's exact finite perfect-information value under a compensatory
additive multi-criteria model. VOIAGE remains the runtime owner. VOP owns the
versioned cross-repository projection and must not convert experimental
repository evidence into a scientific, stable or parity claim.

## Authoritative inputs

- Completed predecessor: `conductor/tracks/specialized-voi-v1-2_20260727/`
  at the branch base; its `specialized-voi-v1.2.0` projection remains unchanged.
- VOIAGE issue #318: <https://github.com/edithatogo/voiage/issues/318>.
- VOIAGE issue #560: <https://github.com/edithatogo/voiage/issues/560>.
- Native delivery subissues #746–#750 under #560.
- Experimental implementation evidence: VOIAGE PR #751,
  <https://github.com/edithatogo/voiage/pull/751>.
- Runtime capability contract:
  `specs/frontier/mcda-information/v1/capabilities.json` in VOIAGE at the PR
  #751 merge revision.

## Requirements

1. M21 is a `Should` requirement planned for v1.3.0 and is not added to or
   relabelled as part of the C16 v1.2.0 contract.
2. The governed estimand retains raw criterion units/directions, fixed ex-ante
   value functions, nonnegative normalized weights and one finite correlated
   outcome/preference law.
3. The result retains baseline and conditional policies, complete ties,
   criterion/preference/joint gross and signed net values, interaction,
   conditional increments, regret, fractional rank acceptability and Pareto
   diagnostics.
4. GitHub hierarchy #313 > #318 > #560 > #746–#750 remains native and
   deduplicated. PR #751 is experimental implementation evidence only.
5. Project 28 exposes MoSCoW `Should`, Contract Version `v1.3.0`, Priority
   `P1`, Status `In Progress`, Lifecycle `Open`, Record ID `mcda-voi`, Track ID
   `supported_frontier_method_completion_20260723`, verified experimental
   evidence and planned synchronization state.
6. Projection synchronization preserves human content, detects three-way
   conflicts, fails closed without credentials or explicit registration, and
   never automatically merges, closes issues or releases artifacts.

## Acceptance criteria

- **AC-01:** Canonical M21, this specification, the projection and the VOIAGE
  requirement agree on `Should` and planned v1.3.0.
- **AC-02:** C16 and `specialized-voi-v1.2.0` retain their completed v1.2.0
  semantics and remain explicit predecessors rather than mutable aliases.
- **AC-03:** Projection records #318, #560, #746–#750 and PR #751 with exact
  track, record and capability references.
- **AC-04:** Mermaid designs preserve conditioning, normalization,
  decomposition and promotion boundaries.
- **AC-05:** Local validation and hosted exact-head checks pass for the C17
  governance revision.
- **AC-06:** Scientific review, stable promotion, Rust/R/Julia parity, release
  and issue closure remain pending unless separately evidenced and authorized.

## External gates

- Independent scientific review of the additive-MCDA estimand and terminology.
- Hosted exact-head validation of the C17 projection and synchronization path.
- Rust, R and Julia shared-fixture parity; Mojo remains external.
- Explicit stable-promotion decision.
- Explicit release and issue-closure authorization.

## Out of scope

- Implementing numerical methods in VOP or duplicating VOIAGE runtime code.
- AHP elicitation, outranking, veto or non-compensatory aggregation.
- Post-information normalization or ordinal labels treated as cardinal weights.
- Imperfect/sample information, endogenous feasible sets or social choice.
- Rewriting C16 evidence or the v1.2.0 projection.
- Automatic merge, issue closure, release, scientific approval or risk acceptance.
