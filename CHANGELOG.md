# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2](https://github.com/edithatogo/vop_poc_nz/compare/v0.2.1...v0.2.2) (2025-12-07)


### Features

* Add 'init' command for Snakemake scaffolding ([9d1283a](https://github.com/edithatogo/vop_poc_nz/commit/9d1283ad7ab29dc104969b52c89c05d7f1de6f0b))
* Add 3-way DSA verification script, update analysis and reporting modules, and introduce .dockerignore. ([b227fe7](https://github.com/edithatogo/vop_poc_nz/commit/b227fe7d9904a12057f7a3103ee3af5843e44905))
* Add CLI with 'run' and 'report' subcommands ([d642c0f](https://github.com/edithatogo/vop_poc_nz/commit/d642c0fb8c99e28971184c0119bac8b337d97a41))
* Add comparative multi-intervention visualizations ([1a9025c](https://github.com/edithatogo/vop_poc_nz/commit/1a9025c488a7292fd2954be93f7e1f928df79730))
* Add comprehensive DSA enhancements ([d0f5eae](https://github.com/edithatogo/vop_poc_nz/commit/d0f5eaea4cff63e45aa486f4128b0d0dd6fbd9e8))
* Add custom Sobol sensitivity analysis (no SALib) ([3613aa9](https://github.com/edithatogo/vop_poc_nz/commit/3613aa964323c6f618afb1b04843a9653885b83d))
* Add discounting to BIA for multi-year analyses ([188796c](https://github.com/edithatogo/vop_poc_nz/commit/188796c16f6f9ad04b2455db72c1fc2f6b5c8f50))
* add extended hypothesis tests, codecov config, and type stubs ([9cc6263](https://github.com/edithatogo/vop_poc_nz/commit/9cc626363b66de39dc79755f54f08f3f37007d18))
* Add extended visualization functions ([bf7eae7](https://github.com/edithatogo/vop_poc_nz/commit/bf7eae721351152d9aa5dbd16e9ec8170d4b4f89))
* Add friction cost method and DCEA results table generation ([43a3ea2](https://github.com/edithatogo/vop_poc_nz/commit/43a3ea2bfd0e4b5f71a24b13cc8616400c35bbd9))
* add mkdocs documentation, release-please, and conda-forge recipe ([fc5f9d7](https://github.com/edithatogo/vop_poc_nz/commit/fc5f9d769c07b8a8d39e8072c9798fc0832d6fb1))
* Add new cost-effectiveness, decision analysis, and equity figures for NZMJ feedback, and update core analysis and utility scripts. ([ac23a3e](https://github.com/edithatogo/vop_poc_nz/commit/ac23a3e90902774a0afc778e1e33ad22bb7c3535))
* Add new visualization module, project documentation, and update extensive test suite. ([069eec8](https://github.com/edithatogo/vop_poc_nz/commit/069eec88dbecc28821fdf6ef575a9d0256dd6ea2))
* Add plot_decision_tree function and call it in main_analysis ([b7618c9](https://github.com/edithatogo/vop_poc_nz/commit/b7618c92af64b6b904089e21fcd9f4d40b7e4e9b))
* Add productivity costs and friction cost params to parameters.yaml ([16a8318](https://github.com/edithatogo/vop_poc_nz/commit/16a8318d069bafe3cddb906b8bd6101f181b1017))
* Add profiling and DSA parameter abstraction ([6f6f554](https://github.com/edithatogo/vop_poc_nz/commit/6f6f55489096d1bb077c25f812ffdfb834496d4c))
* Add realistic NZ population parameters ([33a660f](https://github.com/edithatogo/vop_poc_nz/commit/33a660fef6b6a571dd6e712ae8153639e31ca12e))
* Complete comprehensive DSA enhancement ([464a7a1](https://github.com/edithatogo/vop_poc_nz/commit/464a7a158fd9b2cf38f29b5845c5e1785bdf2ba1))
* enhance CI/CD, achieve 95%+ test coverage, fix linting ([07ca4a9](https://github.com/edithatogo/vop_poc_nz/commit/07ca4a9d3b2b23f2ace587bfab4ccc68c6091f08))
* Finalize project and prepare for submission ([d20db5d](https://github.com/edithatogo/vop_poc_nz/commit/d20db5d7feb683263a865e7f21e6fc18b7b27a67))
* implement comprehensive DSA analysis, add comparative BIA visualization, and adjust cost parameters. ([5e91ef7](https://github.com/edithatogo/vop_poc_nz/commit/5e91ef7e68e7495a340a6b54ad34f1073969504a))
* Refactor and expand core CEA, VOI, DCEA, and DSA analysis, improve visualizations and reporting, and add testing and logging. ([8d0bbed](https://github.com/edithatogo/vop_poc_nz/commit/8d0bbeddcb646d0453660dddf9ee569cd08b8dc8))


### Bug Fixes

* Add initial_population to placeholder parameter functions ([a42060e](https://github.com/edithatogo/vop_poc_nz/commit/a42060ed2dd5cbbc777552b228d7a9d575f0b2fc))
* Add missing dependencies (hypothesis, pandera) to requirements.txt to resolve CI ModuleNotFoundError ([cb02905](https://github.com/edithatogo/vop_poc_nz/commit/cb02905e4a41e271b109df6f8559673f6ba85036))
* Add missing imports to main_analysis.py to resolve 23 ruff linting errors ([e4dadaa](https://github.com/edithatogo/vop_poc_nz/commit/e4dadaa71535f242fa5ba6e481cfd181e23da561))
* add RUF043 and RUF059 to lint ignore list ([c3ba341](https://github.com/edithatogo/vop_poc_nz/commit/c3ba34131642bce9da5187a72c2d40eef38c9d1c))
* Add tabulate dependency for reporting tests ([b881d5f](https://github.com/edithatogo/vop_poc_nz/commit/b881d5fe5a67030e9381dbce84d6db2d971a065b))
* Address linting errors and formatting ([14966a9](https://github.com/edithatogo/vop_poc_nz/commit/14966a92238db0787f8012fbdbd1c261f600c179))
* all 13 CEA model core tests now passing ([1240f30](https://github.com/edithatogo/vop_poc_nz/commit/1240f30c94eb66b66256b204e47d267ebe68be29))
* **ci:** optimize memory usage and enhance CI pipeline ([e7bf1a6](https://github.com/edithatogo/vop_poc_nz/commit/e7bf1a6934f5be5e9c5075cd8571c1d3b09f1d41))
* correct API doc references to existing modules ([0665538](https://github.com/edithatogo/vop_poc_nz/commit/0665538d49da1d3d1664b0294aeaa6b686896218))
* correct module name in docs and fix flaky test ([422eebb](https://github.com/edithatogo/vop_poc_nz/commit/422eebb9de9f833621409066f53b8bc11d2a3c5b))
* Correct NameError for total_gain in run_dcea ([0195927](https://github.com/edithatogo/vop_poc_nz/commit/0195927fe8ec25e4fb4648d490d2751f474346f6))
* Correctly handle discount_rate in subgroup analysis ([b267d58](https://github.com/edithatogo/vop_poc_nz/commit/b267d5871b417788dd3175b16be2de08f0a31359))
* **deps:** restore pydantic, pandera, jinja2, tabulate to runtime dependencies ([6a265d5](https://github.com/edithatogo/vop_poc_nz/commit/6a265d55879ae4610fe0455f0dfcba8a8268861c))
* **docs:** correct relative links in quickstart ([6fb0fab](https://github.com/edithatogo/vop_poc_nz/commit/6fb0fabaf89ad63c53c5112c9a2913d1c77306a3))
* extend deptry ignore rules for optional and transitive dependencies ([aac67bb](https://github.com/edithatogo/vop_poc_nz/commit/aac67bb638f8ae9580f5babc95c69c5ce395c6e3))
* final CI fixes ([5ed5b11](https://github.com/edithatogo/vop_poc_nz/commit/5ed5b1150fea2035aea79ef53b6aba193422a0c6))
* format tests and exclude .tox from deptry ([9d8bf6b](https://github.com/edithatogo/vop_poc_nz/commit/9d8bf6b96023e3d68212bd04428e3f2dc305ef3c))
* Handle list type for intervention_specific_costs in _calculate_friction_cost ([8cbd490](https://github.com/edithatogo/vop_poc_nz/commit/8cbd4901db99da6864780af5d02de597e4f7049f))
* Handle missing discount_rate gracefully in reporting.py and remove debug prints ([27b76d9](https://github.com/edithatogo/vop_poc_nz/commit/27b76d967bf17cab97a7261fe3a036a42bc5bfda))
* ignore RUF043 regex metacharacter warnings in tests ([50fa721](https://github.com/edithatogo/vop_poc_nz/commit/50fa7212901d9f6ea448e2fcbb6241867ef2bd7d))
* Import copy module in reporting.py ([0a4dd6d](https://github.com/edithatogo/vop_poc_nz/commit/0a4dd6de9b162f4ac7f11800856676924950a13e))
* Import plot_decision_tree in main_analysis.py ([0985dbe](https://github.com/edithatogo/vop_poc_nz/commit/0985dbe867178d13c29543201bd390832e603b9d))
* Make EVPI calculation robust against floating point noise and scalar ambiguity ([1ae27c5](https://github.com/edithatogo/vop_poc_nz/commit/1ae27c588fe303550cd1a319d612d9b116c9feee))
* Make graphviz import optional in visualizations.py to resolve CI ImportError ([9983070](https://github.com/edithatogo/vop_poc_nz/commit/99830700725f1f99e2166aa7ceafac92913d12e0))
* Pass discount_rate as argument to run_cea ([049f6fd](https://github.com/edithatogo/vop_poc_nz/commit/049f6fdc4874f3e6150820b57e8ccec5a70b8f0c))
* Preserve discount_rate in subgroup analysis ([bb6fdd7](https://github.com/edithatogo/vop_poc_nz/commit/bb6fdd7623d3eaa31a482374febef9f256792f11))
* Preserve discount_rate in subgroup analysis by explicitly setting it ([1e7a7a0](https://github.com/edithatogo/vop_poc_nz/commit/1e7a7a0dc424c076178713181fef51740e2de5e0))
* PSA visualization and profiling enhancements ([b5c838a](https://github.com/edithatogo/vop_poc_nz/commit/b5c838a38abccca28d940d6b5e7adbf79b173fbd))
* Resolve IndentationError in plotting.py ([791d7ce](https://github.com/edithatogo/vop_poc_nz/commit/791d7ce12a4cf729b406b1917c41a7a540b4fd33))
* Resolve NameError for copy in main_analysis.py ([f865120](https://github.com/edithatogo/vop_poc_nz/commit/f865120f5196af0ee4e9f7817475ba24ecce9f5e))
* Resolve NameError for Dict in plotting.py ([6603686](https://github.com/edithatogo/vop_poc_nz/commit/66036869cf10b37b705cd97819e32e84ba9737f8))
* resolve remaining CI failures ([c12615b](https://github.com/edithatogo/vop_poc_nz/commit/c12615b3f8226c9599eca3317547956ce3d033bc))
* resolve remaining CI lint and pre-commit issues ([e7c211f](https://github.com/edithatogo/vop_poc_nz/commit/e7c211fa1021c85861e7fcba4b6ee77119a99379))
* Restore run_analysis_pipeline function ([0e21817](https://github.com/edithatogo/vop_poc_nz/commit/0e21817983f241a047369336c3d9ea7b06ab2037))
* sync version to 0.1.7 across all files ([4fce93c](https://github.com/edithatogo/vop_poc_nz/commit/4fce93cff9af1aba63145d532777ce5dc17bde75))
* update mkdocs config and release-please workflow ([ddf5b1f](https://github.com/edithatogo/vop_poc_nz/commit/ddf5b1f7ce85be506612e65b1fc49b078e2aad8d))
* update ORCID, add pandera/pydantic deps, fix formatting ([170c549](https://github.com/edithatogo/vop_poc_nz/commit/170c54963d1aa07616fae8bba40fcd95b3dbd973))
* Update relative paths for tutorial and examples links in quickstart guide. ([cf81949](https://github.com/edithatogo/vop_poc_nz/commit/cf819498a8de40d75d168151312bcbd2acf9ba1a))
* Update release workflow to use twine with token secrets ([0a94486](https://github.com/edithatogo/vop_poc_nz/commit/0a944866ec081680e2809aa19f59293398705292))
* update ruff hook id to ruff-check for v0.14+ ([eeb0cbf](https://github.com/edithatogo/vop_poc_nz/commit/eeb0cbfa2d576df1d30b458e0b4c3bdca27478aa))
* Use deepcopy in generate_comprehensive_report to preserve params ([bf98002](https://github.com/edithatogo/vop_poc_nz/commit/bf98002d6c0d81536c1d6627507411544b5e4bfd))
* Use selected_interventions in perform_voi_analysis call ([c19d816](https://github.com/edithatogo/vop_poc_nz/commit/c19d81606bf21ac47e90a61458229a333ced0abd))
* use tox dash prefix to ignore vulture exit code ([1e51da9](https://github.com/edithatogo/vop_poc_nz/commit/1e51da93ce5513e33c0eba5f7023f7aac7c97b0c))


### Documentation

* add FORMULAE.md with mathematical specifications ([2cd31f3](https://github.com/edithatogo/vop_poc_nz/commit/2cd31f37cead719f99399bdb9897658e8ca2a90c))
* Add mermaid architecture diagrams (.mmd) ([00f1f51](https://github.com/edithatogo/vop_poc_nz/commit/00f1f51c60f18758a40985390127e14a3ea8e9b3))
* add Zenodo DOI and OSF links ([5f84fa2](https://github.com/edithatogo/vop_poc_nz/commit/5f84fa2233c7678b24b5f7e0dc046f404ae60266))
* Integrate new Tutorials and Examples pages into the MkDocs site navigation. ([3590627](https://github.com/edithatogo/vop_poc_nz/commit/359062765f417f732c7174d1973cd1114abfc280))
* reframe documentation around Value of Perspective (VoP) concept ([a2e250f](https://github.com/edithatogo/vop_poc_nz/commit/a2e250f2743805f25da1702f31594ee30079557f))
* update README badges to point to actions ([0b9ad22](https://github.com/edithatogo/vop_poc_nz/commit/0b9ad22f6433ae99f2cad9c71c88e6af25e5f343))
* Update README with DCEA focus and new features ([41ec283](https://github.com/edithatogo/vop_poc_nz/commit/41ec283ca90c152795d3cd34b722b1a22ec11d06))


### Code Refactoring

* Consolidate cost-effectiveness plane functions ([e70265e](https://github.com/edithatogo/vop_poc_nz/commit/e70265eca4d45dd9fb9b1c55901640e7c092b72b))


### Tests

* add CEA model core coverage tests (11/13 passing) ([eb92b76](https://github.com/edithatogo/vop_poc_nz/commit/eb92b76f48ab6d7a8b1ca2f65c50ba7cf9263ee3))
* add comprehensive coverage tests for DCEA, Sobol, tables, and visualizations ([f011161](https://github.com/edithatogo/vop_poc_nz/commit/f011161e227ca88ae1bc03b377d6b48ac55b1c54))
* Add DSA enhancement test script ([179a591](https://github.com/edithatogo/vop_poc_nz/commit/179a5914acb3b75886cd32ab96ea4336cf3e7fa5))
* add value of information coverage tests (7/10 passing) ([90f6111](https://github.com/edithatogo/vop_poc_nz/commit/90f61116bec326622b0efafa9463a67a090cd2db))


### CI/CD

* bump actions/checkout from 4 to 6 ([cfd63bc](https://github.com/edithatogo/vop_poc_nz/commit/cfd63bce5f6103c0956547299ca677b46d6ba061))
* bump actions/checkout from 4 to 6 ([c074374](https://github.com/edithatogo/vop_poc_nz/commit/c0743744b54e97e08afaff151b219aeb067f4f75))
* bump actions/setup-python from 5 to 6 ([18303c8](https://github.com/edithatogo/vop_poc_nz/commit/18303c85f652d59b89a01f6e225b12f6b022973f))
* bump actions/setup-python from 5 to 6 ([2c1ba77](https://github.com/edithatogo/vop_poc_nz/commit/2c1ba77a15fead0eb76db91051d9248aedaa4d60))
* bump codecov/codecov-action from 4 to 5 ([945560a](https://github.com/edithatogo/vop_poc_nz/commit/945560a9a5a045cf198cc0474fe741d8b3a12ea1))
* bump codecov/codecov-action from 4 to 5 ([6d831de](https://github.com/edithatogo/vop_poc_nz/commit/6d831de009dee301d913985e9654fd2350455f62))
* fix pre-commit and tox lint issues ([11ecbe5](https://github.com/edithatogo/vop_poc_nz/commit/11ecbe56d5ca348ceb1633300057a5524e176f9e))

## [0.2.1](https://github.com/edithatogo/vop_poc_nz/compare/v0.2.0...v0.2.1) (2025-12-07)


### Features

* Add 'init' command for Snakemake scaffolding ([9d1283a](https://github.com/edithatogo/vop_poc_nz/commit/9d1283ad7ab29dc104969b52c89c05d7f1de6f0b))
* Add 3-way DSA verification script, update analysis and reporting modules, and introduce .dockerignore. ([b227fe7](https://github.com/edithatogo/vop_poc_nz/commit/b227fe7d9904a12057f7a3103ee3af5843e44905))
* Add CLI with 'run' and 'report' subcommands ([d642c0f](https://github.com/edithatogo/vop_poc_nz/commit/d642c0fb8c99e28971184c0119bac8b337d97a41))
* Add comparative multi-intervention visualizations ([1a9025c](https://github.com/edithatogo/vop_poc_nz/commit/1a9025c488a7292fd2954be93f7e1f928df79730))
* Add comprehensive DSA enhancements ([d0f5eae](https://github.com/edithatogo/vop_poc_nz/commit/d0f5eaea4cff63e45aa486f4128b0d0dd6fbd9e8))
* Add custom Sobol sensitivity analysis (no SALib) ([3613aa9](https://github.com/edithatogo/vop_poc_nz/commit/3613aa964323c6f618afb1b04843a9653885b83d))
* Add discounting to BIA for multi-year analyses ([188796c](https://github.com/edithatogo/vop_poc_nz/commit/188796c16f6f9ad04b2455db72c1fc2f6b5c8f50))
* add extended hypothesis tests, codecov config, and type stubs ([9cc6263](https://github.com/edithatogo/vop_poc_nz/commit/9cc626363b66de39dc79755f54f08f3f37007d18))
* Add extended visualization functions ([bf7eae7](https://github.com/edithatogo/vop_poc_nz/commit/bf7eae721351152d9aa5dbd16e9ec8170d4b4f89))
* Add friction cost method and DCEA results table generation ([43a3ea2](https://github.com/edithatogo/vop_poc_nz/commit/43a3ea2bfd0e4b5f71a24b13cc8616400c35bbd9))
* add mkdocs documentation, release-please, and conda-forge recipe ([fc5f9d7](https://github.com/edithatogo/vop_poc_nz/commit/fc5f9d769c07b8a8d39e8072c9798fc0832d6fb1))
* Add new cost-effectiveness, decision analysis, and equity figures for NZMJ feedback, and update core analysis and utility scripts. ([ac23a3e](https://github.com/edithatogo/vop_poc_nz/commit/ac23a3e90902774a0afc778e1e33ad22bb7c3535))
* Add new visualization module, project documentation, and update extensive test suite. ([069eec8](https://github.com/edithatogo/vop_poc_nz/commit/069eec88dbecc28821fdf6ef575a9d0256dd6ea2))
* Add plot_decision_tree function and call it in main_analysis ([b7618c9](https://github.com/edithatogo/vop_poc_nz/commit/b7618c92af64b6b904089e21fcd9f4d40b7e4e9b))
* Add productivity costs and friction cost params to parameters.yaml ([16a8318](https://github.com/edithatogo/vop_poc_nz/commit/16a8318d069bafe3cddb906b8bd6101f181b1017))
* Add profiling and DSA parameter abstraction ([6f6f554](https://github.com/edithatogo/vop_poc_nz/commit/6f6f55489096d1bb077c25f812ffdfb834496d4c))
* Add realistic NZ population parameters ([33a660f](https://github.com/edithatogo/vop_poc_nz/commit/33a660fef6b6a571dd6e712ae8153639e31ca12e))
* Complete comprehensive DSA enhancement ([464a7a1](https://github.com/edithatogo/vop_poc_nz/commit/464a7a158fd9b2cf38f29b5845c5e1785bdf2ba1))
* enhance CI/CD, achieve 95%+ test coverage, fix linting ([07ca4a9](https://github.com/edithatogo/vop_poc_nz/commit/07ca4a9d3b2b23f2ace587bfab4ccc68c6091f08))
* Finalize project and prepare for submission ([d20db5d](https://github.com/edithatogo/vop_poc_nz/commit/d20db5d7feb683263a865e7f21e6fc18b7b27a67))
* implement comprehensive DSA analysis, add comparative BIA visualization, and adjust cost parameters. ([5e91ef7](https://github.com/edithatogo/vop_poc_nz/commit/5e91ef7e68e7495a340a6b54ad34f1073969504a))
* Refactor and expand core CEA, VOI, DCEA, and DSA analysis, improve visualizations and reporting, and add testing and logging. ([8d0bbed](https://github.com/edithatogo/vop_poc_nz/commit/8d0bbeddcb646d0453660dddf9ee569cd08b8dc8))


### Bug Fixes

* Add initial_population to placeholder parameter functions ([a42060e](https://github.com/edithatogo/vop_poc_nz/commit/a42060ed2dd5cbbc777552b228d7a9d575f0b2fc))
* Add missing dependencies (hypothesis, pandera) to requirements.txt to resolve CI ModuleNotFoundError ([cb02905](https://github.com/edithatogo/vop_poc_nz/commit/cb02905e4a41e271b109df6f8559673f6ba85036))
* Add missing imports to main_analysis.py to resolve 23 ruff linting errors ([e4dadaa](https://github.com/edithatogo/vop_poc_nz/commit/e4dadaa71535f242fa5ba6e481cfd181e23da561))
* add RUF043 and RUF059 to lint ignore list ([c3ba341](https://github.com/edithatogo/vop_poc_nz/commit/c3ba34131642bce9da5187a72c2d40eef38c9d1c))
* Add tabulate dependency for reporting tests ([b881d5f](https://github.com/edithatogo/vop_poc_nz/commit/b881d5fe5a67030e9381dbce84d6db2d971a065b))
* Address linting errors and formatting ([14966a9](https://github.com/edithatogo/vop_poc_nz/commit/14966a92238db0787f8012fbdbd1c261f600c179))
* all 13 CEA model core tests now passing ([1240f30](https://github.com/edithatogo/vop_poc_nz/commit/1240f30c94eb66b66256b204e47d267ebe68be29))
* **ci:** optimize memory usage and enhance CI pipeline ([e7bf1a6](https://github.com/edithatogo/vop_poc_nz/commit/e7bf1a6934f5be5e9c5075cd8571c1d3b09f1d41))
* correct API doc references to existing modules ([0665538](https://github.com/edithatogo/vop_poc_nz/commit/0665538d49da1d3d1664b0294aeaa6b686896218))
* correct module name in docs and fix flaky test ([422eebb](https://github.com/edithatogo/vop_poc_nz/commit/422eebb9de9f833621409066f53b8bc11d2a3c5b))
* Correct NameError for total_gain in run_dcea ([0195927](https://github.com/edithatogo/vop_poc_nz/commit/0195927fe8ec25e4fb4648d490d2751f474346f6))
* Correctly handle discount_rate in subgroup analysis ([b267d58](https://github.com/edithatogo/vop_poc_nz/commit/b267d5871b417788dd3175b16be2de08f0a31359))
* extend deptry ignore rules for optional and transitive dependencies ([aac67bb](https://github.com/edithatogo/vop_poc_nz/commit/aac67bb638f8ae9580f5babc95c69c5ce395c6e3))
* final CI fixes ([5ed5b11](https://github.com/edithatogo/vop_poc_nz/commit/5ed5b1150fea2035aea79ef53b6aba193422a0c6))
* format tests and exclude .tox from deptry ([9d8bf6b](https://github.com/edithatogo/vop_poc_nz/commit/9d8bf6b96023e3d68212bd04428e3f2dc305ef3c))
* Handle list type for intervention_specific_costs in _calculate_friction_cost ([8cbd490](https://github.com/edithatogo/vop_poc_nz/commit/8cbd4901db99da6864780af5d02de597e4f7049f))
* Handle missing discount_rate gracefully in reporting.py and remove debug prints ([27b76d9](https://github.com/edithatogo/vop_poc_nz/commit/27b76d967bf17cab97a7261fe3a036a42bc5bfda))
* ignore RUF043 regex metacharacter warnings in tests ([50fa721](https://github.com/edithatogo/vop_poc_nz/commit/50fa7212901d9f6ea448e2fcbb6241867ef2bd7d))
* Import copy module in reporting.py ([0a4dd6d](https://github.com/edithatogo/vop_poc_nz/commit/0a4dd6de9b162f4ac7f11800856676924950a13e))
* Import plot_decision_tree in main_analysis.py ([0985dbe](https://github.com/edithatogo/vop_poc_nz/commit/0985dbe867178d13c29543201bd390832e603b9d))
* Make EVPI calculation robust against floating point noise and scalar ambiguity ([1ae27c5](https://github.com/edithatogo/vop_poc_nz/commit/1ae27c588fe303550cd1a319d612d9b116c9feee))
* Make graphviz import optional in visualizations.py to resolve CI ImportError ([9983070](https://github.com/edithatogo/vop_poc_nz/commit/99830700725f1f99e2166aa7ceafac92913d12e0))
* Pass discount_rate as argument to run_cea ([049f6fd](https://github.com/edithatogo/vop_poc_nz/commit/049f6fdc4874f3e6150820b57e8ccec5a70b8f0c))
* Preserve discount_rate in subgroup analysis ([bb6fdd7](https://github.com/edithatogo/vop_poc_nz/commit/bb6fdd7623d3eaa31a482374febef9f256792f11))
* Preserve discount_rate in subgroup analysis by explicitly setting it ([1e7a7a0](https://github.com/edithatogo/vop_poc_nz/commit/1e7a7a0dc424c076178713181fef51740e2de5e0))
* PSA visualization and profiling enhancements ([b5c838a](https://github.com/edithatogo/vop_poc_nz/commit/b5c838a38abccca28d940d6b5e7adbf79b173fbd))
* Resolve IndentationError in plotting.py ([791d7ce](https://github.com/edithatogo/vop_poc_nz/commit/791d7ce12a4cf729b406b1917c41a7a540b4fd33))
* Resolve NameError for copy in main_analysis.py ([f865120](https://github.com/edithatogo/vop_poc_nz/commit/f865120f5196af0ee4e9f7817475ba24ecce9f5e))
* Resolve NameError for Dict in plotting.py ([6603686](https://github.com/edithatogo/vop_poc_nz/commit/66036869cf10b37b705cd97819e32e84ba9737f8))
* resolve remaining CI failures ([c12615b](https://github.com/edithatogo/vop_poc_nz/commit/c12615b3f8226c9599eca3317547956ce3d033bc))
* resolve remaining CI lint and pre-commit issues ([e7c211f](https://github.com/edithatogo/vop_poc_nz/commit/e7c211fa1021c85861e7fcba4b6ee77119a99379))
* Restore run_analysis_pipeline function ([0e21817](https://github.com/edithatogo/vop_poc_nz/commit/0e21817983f241a047369336c3d9ea7b06ab2037))
* sync version to 0.1.7 across all files ([4fce93c](https://github.com/edithatogo/vop_poc_nz/commit/4fce93cff9af1aba63145d532777ce5dc17bde75))
* update mkdocs config and release-please workflow ([ddf5b1f](https://github.com/edithatogo/vop_poc_nz/commit/ddf5b1f7ce85be506612e65b1fc49b078e2aad8d))
* update ORCID, add pandera/pydantic deps, fix formatting ([170c549](https://github.com/edithatogo/vop_poc_nz/commit/170c54963d1aa07616fae8bba40fcd95b3dbd973))
* Update release workflow to use twine with token secrets ([0a94486](https://github.com/edithatogo/vop_poc_nz/commit/0a944866ec081680e2809aa19f59293398705292))
* update ruff hook id to ruff-check for v0.14+ ([eeb0cbf](https://github.com/edithatogo/vop_poc_nz/commit/eeb0cbfa2d576df1d30b458e0b4c3bdca27478aa))
* Use deepcopy in generate_comprehensive_report to preserve params ([bf98002](https://github.com/edithatogo/vop_poc_nz/commit/bf98002d6c0d81536c1d6627507411544b5e4bfd))
* Use selected_interventions in perform_voi_analysis call ([c19d816](https://github.com/edithatogo/vop_poc_nz/commit/c19d81606bf21ac47e90a61458229a333ced0abd))
* use tox dash prefix to ignore vulture exit code ([1e51da9](https://github.com/edithatogo/vop_poc_nz/commit/1e51da93ce5513e33c0eba5f7023f7aac7c97b0c))


### Documentation

* add FORMULAE.md with mathematical specifications ([2cd31f3](https://github.com/edithatogo/vop_poc_nz/commit/2cd31f37cead719f99399bdb9897658e8ca2a90c))
* Add mermaid architecture diagrams (.mmd) ([00f1f51](https://github.com/edithatogo/vop_poc_nz/commit/00f1f51c60f18758a40985390127e14a3ea8e9b3))
* add Zenodo DOI and OSF links ([5f84fa2](https://github.com/edithatogo/vop_poc_nz/commit/5f84fa2233c7678b24b5f7e0dc046f404ae60266))
* reframe documentation around Value of Perspective (VoP) concept ([a2e250f](https://github.com/edithatogo/vop_poc_nz/commit/a2e250f2743805f25da1702f31594ee30079557f))
* Update README with DCEA focus and new features ([41ec283](https://github.com/edithatogo/vop_poc_nz/commit/41ec283ca90c152795d3cd34b722b1a22ec11d06))


### Code Refactoring

* Consolidate cost-effectiveness plane functions ([e70265e](https://github.com/edithatogo/vop_poc_nz/commit/e70265eca4d45dd9fb9b1c55901640e7c092b72b))


### Tests

* add CEA model core coverage tests (11/13 passing) ([eb92b76](https://github.com/edithatogo/vop_poc_nz/commit/eb92b76f48ab6d7a8b1ca2f65c50ba7cf9263ee3))
* add comprehensive coverage tests for DCEA, Sobol, tables, and visualizations ([f011161](https://github.com/edithatogo/vop_poc_nz/commit/f011161e227ca88ae1bc03b377d6b48ac55b1c54))
* Add DSA enhancement test script ([179a591](https://github.com/edithatogo/vop_poc_nz/commit/179a5914acb3b75886cd32ab96ea4336cf3e7fa5))
* add value of information coverage tests (7/10 passing) ([90f6111](https://github.com/edithatogo/vop_poc_nz/commit/90f61116bec326622b0efafa9463a67a090cd2db))


### CI/CD

* bump actions/checkout from 4 to 6 ([cfd63bc](https://github.com/edithatogo/vop_poc_nz/commit/cfd63bce5f6103c0956547299ca677b46d6ba061))
* bump actions/checkout from 4 to 6 ([c074374](https://github.com/edithatogo/vop_poc_nz/commit/c0743744b54e97e08afaff151b219aeb067f4f75))
* bump actions/setup-python from 5 to 6 ([18303c8](https://github.com/edithatogo/vop_poc_nz/commit/18303c85f652d59b89a01f6e225b12f6b022973f))
* bump actions/setup-python from 5 to 6 ([2c1ba77](https://github.com/edithatogo/vop_poc_nz/commit/2c1ba77a15fead0eb76db91051d9248aedaa4d60))
* bump codecov/codecov-action from 4 to 5 ([945560a](https://github.com/edithatogo/vop_poc_nz/commit/945560a9a5a045cf198cc0474fe741d8b3a12ea1))
* bump codecov/codecov-action from 4 to 5 ([6d831de](https://github.com/edithatogo/vop_poc_nz/commit/6d831de009dee301d913985e9654fd2350455f62))
* fix pre-commit and tox lint issues ([11ecbe5](https://github.com/edithatogo/vop_poc_nz/commit/11ecbe56d5ca348ceb1633300057a5524e176f9e))

## [0.2.0](https://github.com/edithatogo/vop_poc_nz/compare/v0.1.3...v0.2.0) (2025-11-29)


### Features

* add mkdocs documentation, release-please, and conda-forge recipe ([fc5f9d7](https://github.com/edithatogo/vop_poc_nz/commit/fc5f9d769c07b8a8d39e8072c9798fc0832d6fb1))


### Bug Fixes

* Update release workflow to use twine with token secrets ([0a94486](https://github.com/edithatogo/vop_poc_nz/commit/0a944866ec081680e2809aa19f59293398705292))

## [Unreleased]

## [0.1.3] - 2025-11-29

### Added

- **Enhanced Snakefile Template**
  - `configfile` directive for parameters.yaml integration
  - Output versioning with configurable version tag
  - Logging support with `tee` for all rules
  - `discordance_loss.png` output in workflow
  - `clean_all` rule to remove all outputs
  - Test rule with coverage reporting

### Changed

- **Publication Quality Figures**
  - Increased default DPI from 300 to 1200 for all visualizations
  - Updated DSA plots (tornado, heatmaps, 3D surfaces)
  - Updated comparative visualizations (cash flow, ICER ladder, NMB, equity)

## [0.1.2] - 2025-11-29

### Added

- **Project Scaffolding**
  - `vop-poc-nz init` - Initialize project with Snakefile and parameters template
  - `--force` flag to overwrite existing files
  - Bundled Snakefile template for Snakemake workflow integration

## [0.1.1] - 2025-11-29

### Added

- **Command-Line Interface (CLI)**
  - `vop-poc-nz run` - Run full analysis pipeline (CEA, DCEA, VOI, DSA, reporting)
  - `vop-poc-nz report` - Generate reports from previously saved results
  - `--output-dir` / `-o` flag for custom output directory
  - `--skip-reporting` flag to run analysis only
  - `--version` flag to display version

### Fixed

- Fixed missing `validate_psa_results` import in `value_of_information.py`
- Fixed missing `load_parameters` import in `profile_scalability.py`
- Improved pyright/mypy configuration for src-layout compatibility

## [0.1.0] - 2025-11-29

### Added

- **Core CEA Framework**
  - `MarkovModel` class for health state transitions with proper validation
  - `run_cea()` function for cost-effectiveness analysis
  - Support for both health system and societal perspectives
  - Proper discounting of costs and QALYs

- **Distributional Cost-Effectiveness Analysis (DCEA)**
  - `calculate_gini()` - Gini coefficient for inequality measurement
  - `calculate_atkinson_index()` - Atkinson index with configurable inequality aversion
  - `run_dcea()` - Full distributional analysis with equity weighting
  - Lorenz curve and equity impact plane visualizations

- **Value of Information Analysis**
  - `ProbabilisticSensitivityAnalysis` class for Monte Carlo simulation
  - `calculate_evpi()` - Expected Value of Perfect Information
  - `calculate_evppi()` - Expected Value of Partial Perfect Information
  - CEAC and CEAF curve generation

- **Sensitivity Analysis**
  - `run_dsa()` - One-way deterministic sensitivity analysis
  - `run_two_way_dsa()` - Two-way sensitivity analysis
  - `SobolAnalyzer` - Global sensitivity analysis using Sobol indices
  - Tornado diagram visualizations

- **Budget Impact Analysis**
  - `calculate_budget_impact()` - Multi-year budget projections
  - Support for implementation costs and offsets

- **Validation & Quality**
  - Input validation with pandera schemas
  - Property-based testing with Hypothesis
  - 95%+ test coverage
  - Type hints throughout

- **Visualization**
  - Publication-quality figures with matplotlib
  - Optional plotnine support for ggplot2-style graphics
  - Cost-effectiveness planes, acceptability curves, tornado diagrams

### Fixed

- ICER calculation edge cases (zero denominators)
- Transition matrix validation
- Discounting formula corrections

### Security

- Added pip-audit and bandit security scanning

[Unreleased]: https://github.com/edithatogo/vop_poc_nz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edithatogo/vop_poc_nz/releases/tag/v0.1.0
