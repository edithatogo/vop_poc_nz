# TODO (Coverage Focus)

- [ ] Run pytest after changes to confirm current pass (with fail_under=40).
- [ ] Add smoke tests to cover remaining plots in `visualizations.py`: affordability ribbon, tornado/threshold, PSA planes, dashboards, other VOI/equity plots, etc.
- [ ] Add smoke test or pragma for heavy/demo code in `main_analysis.py`; cover minimal execution path if feasible.
- [ ] Smoke tests for `dcea_analysis.py` and `dsa_analysis.py` public functions (or add `# pragma: no cover` to demo blocks).
- [ ] Expand `value_of_information.py` tests (EVPI/EVPPI/report keys) with dummy PSA data.
- [ ] Add test for `cluster_analysis.perform_clustering` if possible.
- [ ] Cover extra branches in `cea_model_core.py` (subgroup/discount logic).
- [ ] After coverage >=95%, restore coverage `fail_under` to 95 in pyproject.
- [ ] Typing: deferred. When ready, rerun mypy (may need targeted ignores to avoid timeouts).
