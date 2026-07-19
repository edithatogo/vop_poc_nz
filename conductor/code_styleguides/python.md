# Python style guide

- Target Python 3.14 and use complete type annotations for public APIs.
- Prefer immutable Pydantic v2 models, frozen dataclasses, enums, protocols and
  narrow mappings over `Any`, untyped dictionaries or forwarding `**kwargs`.
- Use NumPy-style docstrings for public scientific APIs.
- Keep calculation kernels deterministic and free of logging, plotting and
  filesystem side effects.
- Use domain-specific exceptions and preserve exception causes.
- Use Ruff formatting and lint rules as configured by the repository.
- Avoid mutable global state and mutable default arguments.
- Keep imports grouped as standard library, third party, then local package.
- Match canonical `vop_poc_nz` package paths; top-level `src` modules may only
  be temporary deprecation shims.

