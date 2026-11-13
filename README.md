# Value of Perspective — New Zealand Proof of Concept

[![Tests](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/vop_poc_nz/actions/workflows/ci.yml)
[![Documentation](https://github.com/edithatogo/vop_poc_nz/actions/workflows/docs.yml/badge.svg)](https://edithatogo.github.io/vop_poc_nz/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**This repository is a proof of concept and research application, not a second general-purpose health-economic library.** It demonstrates perspective-dependent decisions, distributional cost-effectiveness and uncertainty using New Zealand scenarios.

Reusable decision, perspective, equity and value-of-information capabilities belong in [Voiage](https://github.com/edithatogo/voiage) after independent verification and qualified consumer migration. NZ datasets, clinical/institutional assumptions, case studies and interpretation remain here. The current code has not all been migrated; this ownership statement is not a claim of native execution, numerical certification or production readiness.

See the [provider-promotion plan](conductor/vop-provider-promotion-20261108.md), [Conductor tracks](conductor/tracks.md) and [canonical manifest](conductor/manifest.json). Historical method IDs, outputs and acceptance limitations remain versioned.

## Scope and ecosystem

The proof of concept includes CEA/Markov examples, distributional analysis, VOI, sensitivity analysis, budget-impact projections and reporting. Each capability retains its own scientific and implementation acceptance requirements; a catalogue entry or passing smoke test does not establish validity for every use.

| Concern | Owner and boundary |
|---|---|
| Generic decision, perspective, equity and VOI methods | Voiage; promote one verified provider slice at a time. |
| NZ evidence, scenarios and application interpretation | This proof of concept. |
| Innovation adoption, competition, substitution and hype effects | Innovate, when a qualified application use requires it. |
| Generic simulation time/state/resources/RNG/replay | Kairos; no duplicate simulation engine. |
| General game algorithms | Paignia, the working name of the capability incubated in `new-drug-reimbursement-game`; extraction/publication pending. |

These boundaries introduce no automatic new dependencies. A prescription count, adoption trajectory or strategic payoff requires an explicit valuation/perspective mapping before VOI analysis. Joint uncertainty and draw identity must be preserved.

## Architecture

![Architecture](docs/diagrams/architecture.mmd)

The existing code separates analysis, pipeline orchestration, visualisation and reporting. The diagrams and examples describe the current prototype; the promotion plan defines the transition to provider-backed computation. See [architecture diagrams](docs/diagrams/) and the [tutorial](docs/TUTORIAL.md).

## Existing demonstration capabilities

- CEA and subgroup/perspective comparisons.
- Distributional analysis using Gini/Atkinson indices and equity weights.
- EVPI/EVPPI, population-level information value and acceptability outputs.
- Sobol-style and deterministic sensitivity analyses.
- Multi-year budget-impact examples.
- Tables, plots and CHEERS-oriented reporting templates.

These descriptions are not blanket validation or CHEERS-compliance claims. Existing custom implementations are candidates for independent verification, replacement by qualified providers or explicitly labelled reference use—not a reason to maintain duplicate generic production methods.

## Quick start

```bash
git clone https://github.com/edithatogo/vop_poc_nz.git
cd vop_poc_nz
python -m venv venv
source venv/bin/activate
pip install -e .

snakemake -c1
snakemake -c1 --config version=v2.0
pytest
ruff check src/ tests/
ruff format --check src/ tests/
```

Use the interpreter/dependency versions accepted by the current repository manifests and Conductor evidence. The commands above describe the prototype workflow, not an independently verified current installation receipt.

### Python example

```python
from vop_poc_nz import SobolAnalyzer, run_cea, run_dcea
import yaml

with open("src/parameters.yaml") as f:
    params = yaml.safe_load(f)

results = run_cea(
    params["hpv_vaccination"],
    perspective="societal",
    productivity_cost_method="human_capital",
)

if results["subgroup_results"]:
    equity = run_dcea(
        results["subgroup_results"],
        epsilon=0.5,
        equity_weights={"Low_SES": 1.5, "High_SES": 1.0},
    )
    print(equity)
```

This is a retained prototype API example. Provider migrations must version any changed behaviour and test a clean installed consumer rather than silently redirecting scientific methods.

## Repository and outputs

Historical examples live under `src/`, including `cea_model_core.py`, `dcea_equity_analysis.py`, `value_of_information.py`, `sobol_analysis.py`, `dsa_analysis.py`, `bia_model.py`, pipeline and visualisation modules. Resolve the current canonical package paths before implementation; directory examples are not a substitute for source inspection.

Tests are under `tests/`; documentation and diagrams are under `docs/`; the workflow is in `Snakefile`. Outputs are versioned under `output/`, with the existing `output/latest/` convention.

Typical outputs include perspective-comparison ICER tables, parameter/source tables, distributional results, sensitivity indices, acceptability curves/frontiers, population EVPI, equity plots and report/checklist drafts. Their scientific interpretation depends on the input evidence, selected method and recorded acceptance status.

## Testing and qualification

```bash
pytest
pytest --cov=src --cov-report=html
pytest tests/test_dcea_equity_smoke.py
```

Use exact-commit CI and Conductor receipts for current test results. The former README's historical `64/64` claim is not a current-suite certification. Provider promotion additionally needs independent numerical fixtures, invalid-input tests, actual native dispatch, dependency/artefact identity and installed-consumer evidence. Existing scientific, hosted, parity, release and external-publication gates remain separate.

## Methodological references

- Drummond et al. (2015), *Methods for the Economic Evaluation of Health Care Programmes*.
- Cookson et al. (2017), distributional cost-effectiveness analysis.
- Atkinson (1970), measurement of inequality.
- Claxton et al. (2001), value of information.
- Saltelli et al. (2010), variance-based sensitivity analysis.
- Husereau et al. (2022), CHEERS reporting guidelines.

Refer to method-specific source and validation records for exact implementations and their limitations.

## Contributing and licence

Open changes through a feature branch and pull request. Follow the current Conductor manifest, method ownership and provider-promotion plan. Do not add a duplicate generic solver or call planned functionality accepted merely because its documentation exists.

Apache 2.0; see [LICENSE](LICENSE).

## Citation

Retained citation for the historical proof-of-concept release; use the appropriate version when reproducing a specific analysis:

```bibtex
@software{mordaunt_vop_poc_nz_2026,
  title={Value of Perspective: Quantifying Decision Discordance in Health Economic Evaluation},
  author={Mordaunt, Dylan A.},
  year={2026},
  version={0.2.2},
  doi={10.5281/zenodo.17759624},
  url={https://github.com/edithatogo/vop_poc_nz}
}
```

For questions or issues, open a GitHub issue. Developed as an open research proof of concept for Aotearoa New Zealand health-economic analysis.
