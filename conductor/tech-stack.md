# Technology stack

## Runtime and domain modelling

- Python 3.14 with an observational, non-blocking Python 3.14t lane.
- Pydantic v2 for immutable external specifications, validation and generated
  JSON Schema.
- NumPy and xarray for canonical in-memory numerical structures.
- PyArrow 25 and Polars for schema-bearing interchange and validation.
- Optional JAX/NumPyro and other accelerators behind capability declarations
  and equivalence tests.

## Engineering system

- uv is the dependency and lock authority; Pixi delegates Python resolution to
  uv and supplies reproducible tasks.
- Hatch VCS derives versions from Git tags.
- Ruff, ty and BasedPyright enforce formatting, lint and typing.
- pytest, Hypothesis, coverage, mutation testing, contract/integration/E2E
  tests and fresh-process fixtures verify behaviour.
- Scalene 2.3 profiles scheduled/manual representative workloads.
- GitHub Actions supplies CI, security, dependency, build, release,
  provenance and experimental observation lanes.

