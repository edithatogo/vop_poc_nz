# C18 design — planned v1.3.0 residual frontier

```mermaid
flowchart TD
    Inputs["Declared states, policies, units, conditioning and chronology"] --> Decision["Decision or policy consequences"]
    Risk["Risk and constraints"] --> Decision
    Forecast["Forecast or signal law"] --> Decision
    Sources["Dependent information sources"] --> Decision
    Implementation["Current and counterfactual implementation"] --> Decision
    Model["Model and solution uncertainty"] --> Decision
    Decision --> Values["Gross, net, signed and interaction values"]
    Values --> Local["Event, sequential, social, heterogeneity and sample distributions"]
    Local --> Reconcile["Aggregate identities + deterministic diagnostics"]
```

```mermaid
flowchart LR
    C16["Completed C16 / M14-M17"] --> C18["C18 / M22-M31"]
    C17["C17 / M21 additive MCDA"] -. "sibling v1.3 wave" .-> C18
    C18 --> Parent["VOIAGE #318"]
    Parent --> Wave1["#570 #572 #582"]
    Parent --> Wave2["#593 #594"]
    Parent --> Wave3["#596-#600"]
    Wave1 --> Evidence["Experimental runtime evidence"]
    Wave2 --> Evidence
    Wave3 --> Evidence
    Evidence --> Gates{"Science + hosted + parity + promotion?"}
    Gates -->|"pending"| Experimental["Remain experimental"]
    Gates -->|"separately satisfied"| Eligible["Eligible for governed promotion"]
```
