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
    Manifest[conductor/manifest.json\nC00-C13]
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

## Engineering harness

```mermaid
flowchart TD
    Tag[Signed or reviewed Git tag] --> SCM[SCM dynamic version]
    SCM --> Wheel[Wheel and sdist smoke test]
    Config[Pydantic v2 logging settings] --> Log[Human or JSONL logging]
    Context[Run, track and analysis context] --> Log
    UV[uv.lock and uv commands] --> Fast[PR quality gates]
    Pixi[Pixi cross-platform tasks] --> UV
    Fast --> Types[Ruff + BasedPyright + ty]
    Fast --> Tests[Unit + property + integration + E2E]
    Schedule[Scheduled/manual frontier] --> Profile[Scalene artifacts]
    Schedule --> Mutation[Mutation evidence]
    Schedule --> Audit[Dependency and security audit]
    Schedule --> Experimental[Experimental and free-threaded probes]
    Tests --> Release[Attested release gates]
    Profile --> Decisions[Measured backend decisions]
```

The package metadata and runtime `__version__` resolve from Git through the
build backend. Pixi is a cross-platform task surface and delegates Python
resolution to the same uv lock, preventing two divergent dependency truths.
Expensive or ecosystem-sensitive experiments are explicit evidence lanes and
cannot silently redefine the stable runtime contract.

## Specialized VOI v1.2.0

```mermaid
flowchart LR
    Target["Declared scalar or vector target"] --> Functional["Variance or covariance functional"]
    Prior["Prior and conditioning model"] --> Estimation["EVPPI_var / EVSI_var"]
    Sampling["Sampling model and design"] --> Estimation
    Functional --> Estimation
    Estimation --> Assurance["Estimator uncertainty and assurance"]

    Designs["Evaluated feasible designs"] --> EVSI["Decision EVSI"]
    EVSI --> ENBS["Signed ENBS curve"]
    Cost["Research and opportunity cost"] --> ENBS
    ENBS --> COSS["COSS optimum + tie/boundary state"]
    COSS --> Plot["Accessible plotting inputs"]
    EVSI --> Efficiency["EVSI / EVPI diagnostic"]
    EVPI["Commensurate EVPI"] --> Efficiency

    Utility["Declared utility and wealth/reference state"] --> CurrentEU["Current-policy EU"]
    Utility --> Clairvoyant["Clairvoyant-policy EU"]
    CurrentEU --> EUI["EUI"]
    Clairvoyant --> EUI
    CurrentEU --> Prices["Inverse-utility CEI + BPI/SPI roots"]
    Clairvoyant --> Prices
    EUI --> PPI["Anchored PPI"]
    Clairvoyant --> VoC["VoC presentation under #595"]
```

```mermaid
flowchart TD
    Canonical["C16 canonical v1.2.0 requirements"] --> Projection["Versioned public projection"]
    Projection --> Planner["Three-way conflict-safe sync planner"]
    Planner --> Voiage["edithatogo/voiage managed issue sections"]
    Planner --> Other["Other registered consumer repositories"]
    Voiage --> Hierarchy["#313 > #318 > #571/#595/#619"]
    Hierarchy --> Project["GitHub Project 28"]
    Other --> Project
    Project --> Fields["MoSCoW + Contract Version + Track ID + Record ID + Sync State"]
    Conflict{"Conflict, missing credential, or private data?"}
    Planner --> Conflict
    Conflict -->|yes| Stop["Fail closed; emit reconciliation plan"]
    Conflict -->|no and authorized| Apply["Update managed fields only"]
```

## Typed domain and governance model

```mermaid
flowchart LR
    Concern --> Assumption
    Concern --> Risk
    Assumption --> Risk
    Evidence --> Assumption
    Evidence --> Risk
    Evidence --> Decision
    Risk --> Decision
    Decision --> IssueLink
    IssueLink --> GitHubIssue
    GitHubIssue --> Project

    AnalysisSpec --> NumericalPolicy
    AnalysisSpec --> Kernel
    Kernel --> BackendCapabilities
    Kernel --> AnalysisResult
    RunContext --> AnalysisResult
    Evidence --> AnalysisResult
    AnalysisResult --> ArrowContract
```

```mermaid
sequenceDiagram
    participant R as Local registry
    participant P as Sync planner
    participant B as Last sync base
    participant G as GitHub issue/project
    R->>P: Canonical public projection
    B->>P: Base digest
    G->>P: Remote managed projection
    P->>P: Three-way comparison
    alt Conflict or private record
      P-->>R: Fail closed with reconciliation plan
    else Clean dry-run
      P-->>R: Emit deterministic mutation plan
    else Clean and explicitly authorized
      P->>G: Update managed fields only
      G-->>B: Persist new sync base locally
    end
```
