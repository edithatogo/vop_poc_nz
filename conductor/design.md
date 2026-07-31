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
    Manifest[conductor/manifest.json\nC00-C19]
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

## VOP assurance and maintenance v1.3.0

```mermaid
flowchart TD
    Root["C19 / #55 hardening programme"]
    CI["#53 testing and CI"]
    Governance["#54 security and governance"]
    Root --> CI
    Root --> Governance
    CI --> Quality["#60 imports, coverage and static ratchets"]
    CI --> Dependencies["#61 Renovate authority"]
    CI --> Artifacts["#62 generated-artifact hygiene"]
    Governance --> Rules["#57 main ruleset and Actions policy"]
    Governance --> Receipts["#58 immutable dispatch receipts"]
    Governance --> Drift["#59 governance drift validation"]
    Rules --> Evidence["Hosted drift evidence"]
    Receipts --> Evidence
    Quality --> Gates["Exact-head quality gates"]
    Dependencies --> Gates
    Artifacts --> Gates
    Drift --> Project["Project 28 / v1.3.0"]
    Evidence --> Project
    Gates --> Project
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

## Specialized VOI v1.3.0 additive MCDA continuation

```mermaid
flowchart LR
    Raw["Raw criteria with units + directions"] --> Value["Fixed ex-ante value functions"]
    Weights["Nonnegative normalized weights"] --> Additive["Finite additive value model"]
    Joint["Correlated outcome/preference law"] --> Additive
    Value --> Additive
    Additive --> Baseline["Baseline expected policy + complete ties"]
    Joint --> Criterion["Criterion-information partition"]
    Joint --> Preference["Preference-information partition"]
    Joint --> JointInfo["Joint-information partition"]
    Baseline --> Compare["Gross + signed net information value"]
    Criterion --> Compare
    Preference --> Compare
    JointInfo --> Compare
    Compare --> Decomposition["Interaction + conditional increments"]
    Compare --> Diagnostics["Regret + rank acceptability + Pareto"]
```

```mermaid
flowchart TD
    C16["Completed C16 / v1.2.0 predecessor"] --> C17["C17 / M21 planned v1.3.0"]
    C17 --> Projection["Versioned v1.3.0 fail-closed projection"]
    Projection --> Parent["VOIAGE #318 frontier programme"]
    Parent --> MCDA["#560 additive MCDA information value"]
    MCDA --> Children["#746–#750 native delivery subissues"]
    MCDA --> Evidence["PR #751 experimental repository evidence"]
    Evidence --> Gates{"Promotion gates satisfied?"}
    Gates -->|"no"| Experimental["Remain experimental"]
    Gates -->|"scientific + hosted + parity + stable review"| Promotion["Separate promotion decision"]
    Projection --> Project["Project 28: Should + v1.3.0 + In Progress"]
```

## Typed domain and governance model

## Specialized VOI v1.3.0 residual frontier

```mermaid
flowchart TD
    C16["Completed C16 / M14-M17"] --> C18["C18 / M22-M32"]
    C17["C17 / M21 additive MCDA"] -. "parallel v1.3 wave" .-> C18
    C18 --> Parent["VOIAGE #318"]
    Parent --> Decision["Risk, forecasts, source portfolios"]
    Parent --> Decompose["Implementation, model uncertainty, heterogeneity"]
    Parent --> Dynamic["Event, sequential, social and realized sample value"]
    Parent --> Acquisition["#850 / M32 sampling-acquisition-harm research scope"]
    Decision --> Evidence["Experimental runtime evidence"]
    Decompose --> Evidence
    Dynamic --> Evidence
    Acquisition --> Evidence
    Evidence --> Gates{"Scientific + hosted + parity + promotion gates"}
    Gates -->|"pending"| Hold["Remain experimental"]
    Gates -->|"separately satisfied"| Promote["Eligible for governed promotion"]
```

```mermaid
flowchart LR
    Design["Sampling action d and explicit no-sampling comparator d0"] --> Joint["Information and acquisition-harm laws, parties, timing and units"]
    Joint --> ScalarGate{"Mutually exclusive, separable and commensurate?"}
    ScalarGate -->|"Yes"| Scalar["Signed incremental value with harm counted once"]
    ScalarGate -->|"No"| NonScalar["Joint-welfare, constrained or vector result"]
    Scalar --> Candidate["Exact candidate and mathematical feasibility"]
    NonScalar --> Candidate
    Candidate --> Review["Candidate-bound independent scientific/domain review"]
    Review --> Human["Separate named human verdict"]
    Human --> Runtime{"Runtime implementation separately authorized?"}
    Runtime -->|"No or pending"| Unsupported["M32 unsupported research scope"]
    Runtime -->|"Yes"| Future["Future VOIAGE Rust-authoritative implementation"]
    Candidate -. "real-study deployment only" .-> Ethics["Accountable ethics/regulatory authorization where applicable"]
```

```mermaid
flowchart LR
    States["States + probabilities"] --> Policy["Declared policies and feasible sets"]
    Signals["Signals, sources and chronology"] --> Policy
    Uptake["Current and counterfactual implementation"] --> Policy
    Risk["Utility, risk and constraints"] --> Policy
    Policy --> Matrix["Conditional value matrix"]
    Matrix --> Components["Gross, net, signed and interaction components"]
    Components --> Diagnostics["Identities, uncertainty and solver assurance"]
```

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
