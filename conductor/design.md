# VOP–VOIAGE design

## System context

```mermaid
flowchart LR
    Owner[Owner and reviewers]
    Evidence[Evidence and case inputs]
    VOP[VOP research and policy analysis]
    Contract[Versioned VOP–VOIAGE contract]
    Arrow[Arrow IPC and Parquet fixtures]
    VOIAGE[VOIAGE production method API]
    Harness[Cross-repository harness]
    CI[Hosted CI and dependency frontier]
    Publication[Release and publication gates]
    Private[Private .conductor/local state]

    Evidence --> VOP
    VOP --> Contract
    Contract --> Arrow
    Arrow --> VOIAGE
    VOP --> Harness
    VOIAGE --> Harness
    Harness --> CI
    CI --> Publication
    Owner --> Publication
    Private -. never promoted implicitly .-> VOP
```

VOP owns the canonical compatibility policy. VOIAGE pins a byte-identical
mirror with an upstream commit and digest. Both producers embed contract,
schema, method, producer, and interchange metadata in Arrow outputs.

## Conductor and GitHub control plane

```mermaid
flowchart TD
    Manifest[conductor/manifest.json\nC00-C12]
    Tracks[Current track specifications]
    Legacy[VOP legacy track map]
    VoiageArchive[VOIAGE archived track registry]
    Requirements[MoSCoW requirements]
    Design[Mermaid design]
    Issues[GitHub track and ledger issues]
    Project[GitHub Project roadmap]
    PRs[Implementation pull requests]
    Checks[CI, security, quality and benchmark checks]

    Manifest --> Tracks
    Requirements --> Tracks
    Design --> Tracks
    Tracks --> Issues
    Legacy --> Issues
    VoiageArchive --> Issues
    Issues --> Project
    PRs --> Project
    PRs --> Checks
    Checks --> Project
```

Current track issues use a hidden `vop-voiage-conductor-track-id` marker for
idempotent synchronization. Historical ledgers are closed issues with stable
markers and exhaustive track lists. Local files remain the detailed source of
truth; the GitHub Project is the coordination and audit view.

## Interchange sequence

```mermaid
sequenceDiagram
    participant P as Producer
    participant S as Canonical schema
    participant A as Arrow writer
    participant M as Manifest
    participant C as Consumer
    participant G as CI gate

    P->>S: Build ordered logical fields
    S->>S: Canonical JSON SHA-256 fingerprint
    P->>A: Attach contract and method metadata
    A->>M: Write IPC/Parquet and hashes
    C->>M: Verify artifact and contract versions
    C->>A: Read through PyArrow or Polars
    C->>G: Compare schema, metadata and values
    G-->>P: Pass or fail closed
```

## Safety boundaries

- Cross-repository integration is data/contract based, never an implicit
  source import.
- Experimental backends are opt-in and must preserve numerical parity.
- Private files, credentials, unpublished evidence, and owner decisions remain
  outside public issue bodies and promoted artifacts.
- Publication and registry state are independent from implementation state.
