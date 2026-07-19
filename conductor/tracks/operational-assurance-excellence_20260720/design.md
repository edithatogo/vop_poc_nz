# C15 design: Operational Assurance Excellence

```mermaid
flowchart LR
    Source["VOP contracts + scientific oracles"] --> Bundle["Content-addressed bundle"]
    Bundle --> Matrix["VOIAGE current / N-1 / incompatible matrix"]
    Bundle --> Draft["Private signed-release staging"]

    subgraph Quality["Cohort-aware quality"]
        DiffCov["Changed-line + critical coverage"] --> Evidence["Evidence manifest"]
        Mutation["Mutation cohort + debt density"] --> Evidence
        Perf["Repeated samples + confidence interval"] --> Evidence
    end

    subgraph Repro["Independent reproducibility"]
        Linux["Linux normalized build"] --> Compare["Exact digest comparator"]
        Windows["Windows normalized build"] --> Compare
        Compare --> Evidence
    end

    subgraph Telemetry["Collector privacy boundary"]
        App["Correlated redacted spans"] --> Collector["Ephemeral OTel collector"]
        Collector --> Scan["Export privacy + correlation scan"]
        Scan --> Evidence
    end

    Evidence --> Drift["Read-only governance audit"]
    Drift --> Review{"Authorized review"}
    Review -->|approved only| Trusted["Trusted baseline / merge / publication"]
```

## Invariants

1. Cohort identity is derived from source and configuration digests, never labels.
2. Coverage and performance comparisons are fail-closed on missing or malformed
   evidence.
3. Cross-runner comparison normalizes timestamps and platform-specific archive
   metadata before comparing the intended publishable bytes.
4. The collector receives already-redacted telemetry; post-export scanning is a
   second independent privacy control.
5. Approval-bearing operations are staged but not executed by pull-request CI.
