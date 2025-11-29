# Quick Start

This guide walks you through running your first cost-effectiveness analysis with VoP-PoC-NZ.

## Basic CEA Model

```python
from vop_poc_nz import CEAModel, Intervention

# Define interventions
intervention = Intervention(
    name="New Treatment",
    cost=10000,
    qalys=5.2,
    cost_comparator=5000,
    qalys_comparator=4.0,
)

# Create and run model
model = CEAModel(
    interventions=[intervention],
    threshold=50000,  # NZD per QALY
)

results = model.run()
print(results.summary())
```

## Running PSA

```python
from vop_poc_nz import run_psa

# Run probabilistic sensitivity analysis
psa_results = run_psa(
    model,
    n_iterations=1000,
    seed=42,
)

# View results
print(psa_results.mean_icer)
print(psa_results.probability_cost_effective)
```

## Generating Reports

```python
from vop_poc_nz import generate_report

# Create comprehensive report
generate_report(
    results,
    psa_results,
    output_path="analysis_report.html",
    format="html",
)
```

## Snakemake Pipeline

For full reproducible analysis:

```bash
# Run complete analysis pipeline
snakemake --cores 4

# Generate specific outputs
snakemake output/cea_results.csv
```

## Next Steps

- [API Reference](../api/cea_model.md) - Detailed documentation
- [Tutorial Notebook](../tutorials/basic_analysis.md) - Step-by-step guide
- [Examples](../examples/index.md) - Real-world use cases
