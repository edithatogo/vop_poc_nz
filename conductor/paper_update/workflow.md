# Paper-update workflow

```mermaid
flowchart LR
  A[Source and literature agent] --> M[Methods agent]
  B[Reproducibility agent] --> I[Integrator]
  C[Editorial agent] --> I
  M --> I
  I --> V{Contract validator}
  V -- fail --> R[Revise and rerun]
  V -- pass --> P[Human author submission gate]
```

The workflow is fail-closed. It requires the exact VOP release revision,
Sourceright and Authentext tool receipts, claim/evidence mappings, generated
PDF/source hashes, and a human author decision before any external submission.

AuthenText is installed outside this repository and runs only concrete
editorial heuristics over TeX-extracted prose (with commands, math, and
citation keys masked). Its similarity/self-critique output is an editorial
receipt only, never evidence of factual or citation correctness.
