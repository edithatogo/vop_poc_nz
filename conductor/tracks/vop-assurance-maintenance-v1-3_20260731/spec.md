# VOP repository assurance and maintenance closure

## Overview

C19 is the canonical planned-v1.3.0 closure programme for the remaining VOP
repository-policy, cross-repository synchronization, maintainability,
dependency-automation, artifact-hygiene, and governance-drift work. It refines
C07, C10, C12, and C16 without rewriting their completed evidence.

## Requirements

1. Protect `main` with stable required checks and a solo-maintainer-compatible
   recovery path; use read-only GitHub Actions defaults and narrowly scoped job
   permissions.
2. Bind every governance dispatch to a full canonical commit SHA, projection
   digest, correlation ID, consumer receipt, and idempotency key.
3. Move first-party tooling off legacy `src.*` shims and reconcile coverage,
   Ruff, BasedPyright, and `ty` through explicit non-regressing ratchets.
4. Make Renovate the sole dependency-update authority after validation, and
   retain lockfile, frontier, security, and hosted-test evidence.
5. Remove transient debug/build bulk from Git in normal recoverable commits;
   retain only dispositioned, manifest-backed generated artifacts.
6. Validate `AGENTS.md`, the track registry, Mermaid design, issue backlog, and
   hosted settings against canonical sources so governance drift fails closed.

## Acceptance criteria

- **AC-01:** GitHub rules and Actions defaults match M33 and have a
  machine-readable hosted drift receipt.
- **AC-02:** A dispatched projection and consumer outcome are correlated and
  cryptographically bound as specified by M34.
- **AC-03:** Canonical imports and transparent static/coverage ratchets satisfy
  M35 without concealing baseline debt.
- **AC-04:** Renovate authority and artifact dispositions satisfy M36 with no
  unreviewed auto-merge or implicit history rewrite.
- **AC-05:** Canonical manifest, requirements, design, track registry, backlog,
  GitHub hierarchy, and Project 28 fields agree under M37.
- **AC-06:** GitHub hierarchy #55 > #53/#54 > #57–#62 remains native,
  deduplicated, and evidence-linked.

## Non-functional constraints

- Required checks must be named from observed hosted contexts, not guessed.
- A solo maintainer must not be blocked by a second-human approval requirement.
- Credentials and receipts must not expose token material or private evidence.
- Expensive checks remain scheduled/manual where appropriate; PR gates stay
  bounded and deterministic.
- Compatibility removal follows warnings, tests, and a declared version/date.

## External gates

Hosted ruleset changes, repository-wide Actions settings, Renovate activation,
consumer-repository receipt support, merges, issue closure, and releases remain
separate external gates. Repository planning does not mark them complete.

## Out of scope

- Rewriting Git history to purge existing large objects.
- Automatically merging dependency or governance proposals.
- Closing parent issues solely because planning artifacts exist.
- Replacing VOIAGE numerical implementation ownership.

## Authoritative inputs

- `conductor/manifest.json` at the C19 initialization commit.
- `conductor/requirements.md` M33–M37.
- `conductor/design.md` VOP assurance v1.3.0 control plane.
- GitHub issues #55, #53, #54, and native subissues #57–#62.
- GitHub Project 28.
