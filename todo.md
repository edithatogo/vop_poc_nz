# TODO (Coverage Focus)

- [x] Run pytest after changes to confirm current pass (with fail_under=40). `python -m pytest -q` passed on 2025-11-29 (184 tests).
- [x] Add smoke tests to cover remaining plots in `visualizations.py`: affordability ribbon, tornado/threshold, PSA planes, dashboards, other VOI/equity plots, etc. Covered by `tests/test_visualizations_new.py` and `tests/test_cluster_and_dsa_visuals.py`.
- [x] Add smoke test or pragma for heavy/demo code in `main_analysis.py`; cover minimal execution path if feasible. Exercised via `tests/test_smoke_coverage.py` and `tests/test_reporting_and_core.py`.
- [x] Smoke tests for `dcea_analysis.py` and `dsa_analysis.py` public functions (or add `# pragma: no cover` to demo blocks). Functions hit through `tests/test_dcea_*` and `tests/test_coverage_boost.py`.
- [x] Expand `value_of_information.py` tests (EVPI/EVPPI/report keys) with dummy PSA data. Verified by `tests/test_voi_coverage.py` plus supporting smoke/property tests.
- [x] Add test for `cluster_analysis.perform_clustering` if possible. Implemented in `tests/test_core_functions.py` and related coverage suites.
- [x] Cover extra branches in `cea_model_core.py` (subgroup/discount logic). Addressed in `tests/test_cea_model_core_coverage.py` and integration tests.
- [x] After coverage >=95%, restore coverage `fail_under` to 95 in pyproject. `pyproject.toml` now enforces `fail_under = 95`.
- [x] Typing: mypy/pyright configured for src-layout. Pyright passes (0 errors, 17 optional import warnings).

## Release Checklist (2025-11-29) ✅ COMPLETE

- [x] Build distribution (`hatch build` → `vop_poc_nz-0.1.0-py3-none-any.whl`, `.tar.gz`)
- [x] Verify with twine (`twine check dist/*` → PASSED)
- [x] Publish to TestPyPI → https://test.pypi.org/project/vop-poc-nz/0.1.0/
- [x] Smoke test TestPyPI install → Import verified
- [x] Publish to PyPI → https://pypi.org/project/vop-poc-nz/0.1.0/
- [x] Create GitHub release (tag `v0.1.0`) → https://github.com/edithatogo/vop_poc_nz/releases/tag/v0.1.0
