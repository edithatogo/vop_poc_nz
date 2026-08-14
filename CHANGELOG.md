# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

* Tag-derived Hatch VCS versions and runtime metadata resolution.
* Pydantic v2 structured logging with JSONL, run IDs, and bound context.
* Locked Pixi/uv task parity, BasedPyright plus `ty`, Scalene profiling,
  mutation/security/experimental evidence lanes, and SHA-pinned Actions.

## [0.2.4](https://github.com/edithatogo/vop_poc_nz/compare/v0.2.3...v0.2.4) (2026-08-14)


### Features

* add fail-closed paper evidence validator ([a58eee5](https://github.com/edithatogo/vop_poc_nz/commit/a58eee5ecbe705e60879d551feacd97209544bf4))
* add paper update agent and tool contracts ([ab19f28](https://github.com/edithatogo/vop_poc_nz/commit/ab19f28ace33d3b36a11df7ba64dab45fe678536))
* **governance:** dispatch bounded C16 consumer sync ([9190534](https://github.com/edithatogo/vop_poc_nz/commit/9190534328b8e5d5a4d8ab5bbfea21abdba10a5b))
* modernize VOP LaTeX manuscript workflow ([9305c06](https://github.com/edithatogo/vop_poc_nz/commit/9305c067bd41651768161e688a0eda19ed6d5188))
* populate paper evidence receipts and manifest ([b883135](https://github.com/edithatogo/vop_poc_nz/commit/b883135adf14afdbe300a754f8eedafc2a09641e))
* record explicit administrator governance bypass ([dae9d1c](https://github.com/edithatogo/vop_poc_nz/commit/dae9d1cc494f8d7f21d55d00fb25b464e57bfc64))


### Bug Fixes

* align public concern body taxonomy ([c56b32e](https://github.com/edithatogo/vop_poc_nz/commit/c56b32ef603728f0f195cf564ae6fb5792e2ee23))
* align tracked concern Project taxonomy ([d3309f0](https://github.com/edithatogo/vop_poc_nz/commit/d3309f0169cf0b8f1f715c67d288af2d441995e6))
* **conductor:** preserve residual MoSCoW priorities ([#70](https://github.com/edithatogo/vop_poc_nz/issues/70)) ([4c8ae85](https://github.com/edithatogo/vop_poc_nz/commit/4c8ae85deaca82041589b0a664802f0c894648a5))
* validate GitHub workflow path metadata ([f2ede64](https://github.com/edithatogo/vop_poc_nz/commit/f2ede645d0d6ca92202010713a6608845d8b83ff))


### Documentation

* **conductor:** add canonical C18/M32 sampling-harm scope ([#71](https://github.com/edithatogo/vop_poc_nz/issues/71)) ([e0ff1d2](https://github.com/edithatogo/vop_poc_nz/commit/e0ff1d2ce3361d52ee22bb01e105b92653ed606c))
* **conductor:** record C16 mirror merge ([#56](https://github.com/edithatogo/vop_poc_nz/issues/56)) ([23287f8](https://github.com/edithatogo/vop_poc_nz/commit/23287f879fc39974be0fdf5c7a5db2c494dc13aa))
* **conductor:** sync estimation variance C16 evidence ([#64](https://github.com/edithatogo/vop_poc_nz/issues/64)) ([cedc6fb](https://github.com/edithatogo/vop_poc_nz/commit/cedc6fbb17a5d999cb12bb300a01f87d976ec02e))
* **conductor:** sync expected-utility VoC C16 plan ([#66](https://github.com/edithatogo/vop_poc_nz/issues/66)) ([9c25057](https://github.com/edithatogo/vop_poc_nz/commit/9c250572750b5ae760ef8bc1e52b8060a9758d56))
* **conductor:** sync study efficiency C16 evidence ([#65](https://github.com/edithatogo/vop_poc_nz/issues/65)) ([ac61bb9](https://github.com/edithatogo/vop_poc_nz/commit/ac61bb9f46136dad480379785641bdf381ade7c2))
* pin paper tool integration evidence ([56bd7e4](https://github.com/edithatogo/vop_poc_nz/commit/56bd7e41ae81175b2c7ae004bde626865fa29862))
* record verified AuthenText integration ([dcf3605](https://github.com/edithatogo/vop_poc_nz/commit/dcf36050082128dd7b9b0c269a9fec4093a6d349))


### Tests

* assert promoted baseline trust ([b7879bc](https://github.com/edithatogo/vop_poc_nz/commit/b7879bcff3da911b256c716474d0187e34a716f4))


### CI/CD

* verify pinned paper tooling in manuscript build ([2c46db2](https://github.com/edithatogo/vop_poc_nz/commit/2c46db2fe5f907d894bb07f1127c008f10ee462e))

## [0.2.3](https://github.com/edithatogo/vop_poc_nz/compare/v0.2.2...v0.2.3) (2026-07-20)


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
* **assurance:** stage reviewed governance and contract releases ([f000c2c](https://github.com/edithatogo/vop_poc_nz/commit/f000c2c5340d981d218d55eb4b941fe4e64d1319))
* **c14:** add independent analytical references ([3c841cd](https://github.com/edithatogo/vop_poc_nz/commit/3c841cd054666cf3d9ebefd962fc2ea07c20a510))
* **c14:** attest reproducible supply chain ([8966c34](https://github.com/edithatogo/vop_poc_nz/commit/8966c348fde179f644e225a2b6420b8c8b366dac))
* **c14:** audit GitHub governance drift read-only ([98f4de8](https://github.com/edithatogo/vop_poc_nz/commit/98f4de8ca8000c07b42d964bdc0df374e4e041bd))
* **c14:** correlate and redact structured analysis logs ([b6b5b4a](https://github.com/edithatogo/vop_poc_nz/commit/b6b5b4a3488fb43842843ca0416c311b2c3c4234))
* **c14:** enforce dependency and performance frontiers ([e33a890](https://github.com/edithatogo/vop_poc_nz/commit/e33a890586eff25a1ad73588b093de6651c5d20c))
* **c14:** publish deterministic contract bundle ([edd025e](https://github.com/edithatogo/vop_poc_nz/commit/edd025e06427afa624554f8dd8193448759e522a))
* **c15:** add cross-platform operational assurance ([e6ad9fa](https://github.com/edithatogo/vop_poc_nz/commit/e6ad9fae5d02e8384d8750f39c0e10bed0dd0de1))
* Complete comprehensive DSA enhancement ([464a7a1](https://github.com/edithatogo/vop_poc_nz/commit/464a7a158fd9b2cf38f29b5845c5e1785bdf2ba1))
* **contract:** embed shared Arrow metadata ([ba50a6d](https://github.com/edithatogo/vop_poc_nz/commit/ba50a6d9604c5cf3b18bf9fc642650bffc09de69))
* **contract:** publish VOP-VOIAGE compatibility policy ([324640b](https://github.com/edithatogo/vop_poc_nz/commit/324640b36d8911d0bdd1421fd9c67c4ccf9c9173))
* **domain:** add typed CEA contract boundary ([2566c0c](https://github.com/edithatogo/vop_poc_nz/commit/2566c0c26eead086d8ebe74b7cff445f00635ac2))
* enhance CI/CD, achieve 95%+ test coverage, fix linting ([07ca4a9](https://github.com/edithatogo/vop_poc_nz/commit/07ca4a9d3b2b23f2ace587bfab4ccc68c6091f08))
* Finalize project and prepare for submission ([d20db5d](https://github.com/edithatogo/vop_poc_nz/commit/d20db5d7feb683263a865e7f21e6fc18b7b27a67))
* **governance:** add canonical C13 registry ([d7cd1a9](https://github.com/edithatogo/vop_poc_nz/commit/d7cd1a9b0b4ffde7b16c019bcc10afe2b4312f36))
* **governance:** add conflict-safe sync planner ([55b3e5b](https://github.com/edithatogo/vop_poc_nz/commit/55b3e5b1dfb20a4bd58a9155300402e0fb47586d))
* **governance:** add typed concern contracts ([51a6134](https://github.com/edithatogo/vop_poc_nz/commit/51a61340dde73868c319f97a7ce436b0650255be))
* **governance:** register C13 issue projection ([cb59b82](https://github.com/edithatogo/vop_poc_nz/commit/cb59b82a7b2a6061851152d2b414db25c8db7f4f))
* implement comprehensive DSA analysis, add comparative BIA visualization, and adjust cost parameters. ([5e91ef7](https://github.com/edithatogo/vop_poc_nz/commit/5e91ef7e68e7495a340a6b54ad34f1073969504a))
* **interchange:** add cross-repo Arrow compatibility harness ([b6411a8](https://github.com/edithatogo/vop_poc_nz/commit/b6411a89a027703385da86db93d5a762f4a21068))
* **interchange:** adopt latest Arrow-first Python 3.14 stack ([cb5adc3](https://github.com/edithatogo/vop_poc_nz/commit/cb5adc32e8af50e32249649a033a25bbcfd55a24))
* **perspective:** enforce directional current-information EVoP ([02271a2](https://github.com/edithatogo/vop_poc_nz/commit/02271a26ccbc5db4512a2b1a5dfdf4e5120f05b5))
* **pipeline:** add immutable typed analysis boundary ([bb3bda9](https://github.com/edithatogo/vop_poc_nz/commit/bb3bda9b485ff1732c562369fbe49ad99222e658))
* **platform:** add observable frontier harness ([3a9d01f](https://github.com/edithatogo/vop_poc_nz/commit/3a9d01f626ce383c28b88944706a987186b468db))
* Refactor and expand core CEA, VOI, DCEA, and DSA analysis, improve visualizations and reporting, and add testing and logging. ([8d0bbed](https://github.com/edithatogo/vop_poc_nz/commit/8d0bbeddcb646d0453660dddf9ee569cd08b8dc8))


### Bug Fixes

* Add initial_population to placeholder parameter functions ([a42060e](https://github.com/edithatogo/vop_poc_nz/commit/a42060ed2dd5cbbc777552b228d7a9d575f0b2fc))
* Add missing dependencies (hypothesis, pandera) to requirements.txt to resolve CI ModuleNotFoundError ([cb02905](https://github.com/edithatogo/vop_poc_nz/commit/cb02905e4a41e271b109df6f8559673f6ba85036))
* Add missing imports to main_analysis.py to resolve 23 ruff linting errors ([e4dadaa](https://github.com/edithatogo/vop_poc_nz/commit/e4dadaa71535f242fa5ba6e481cfd181e23da561))
* add RUF043 and RUF059 to lint ignore list ([c3ba341](https://github.com/edithatogo/vop_poc_nz/commit/c3ba34131642bce9da5187a72c2d40eef38c9d1c))
* Add tabulate dependency for reporting tests ([b881d5f](https://github.com/edithatogo/vop_poc_nz/commit/b881d5fe5a67030e9381dbce84d6db2d971a065b))
* Address linting errors and formatting ([14966a9](https://github.com/edithatogo/vop_poc_nz/commit/14966a92238db0787f8012fbdbd1c261f600c179))
* all 13 CEA model core tests now passing ([1240f30](https://github.com/edithatogo/vop_poc_nz/commit/1240f30c94eb66b66256b204e47d267ebe68be29))
* **assurance:** bind governance and draft authority ([0b1a507](https://github.com/edithatogo/vop_poc_nz/commit/0b1a5071412a7d935e5b08f84c8b5525afde7c1d))
* **assurance:** close authority and draft races ([b629d29](https://github.com/edithatogo/vop_poc_nz/commit/b629d29a22b2ca03b20b1e0e4d0140e732dc7206))
* **c13:** close principal audit findings ([e26ad8d](https://github.com/edithatogo/vop_poc_nz/commit/e26ad8dbf89244df02e20490bdfc0a08abcee50c))
* **c13:** close typed contract review findings ([0a8cd21](https://github.com/edithatogo/vop_poc_nz/commit/0a8cd21af98c149959835fcf9b5a883080467d28))
* **c13:** harden typed contracts and governance sync ([7faddc8](https://github.com/edithatogo/vop_poc_nz/commit/7faddc8d68f2a165e2140bfd5fa852940b8e7c57))
* **c14:** anchor migration and toolchain assurance ([c174907](https://github.com/edithatogo/vop_poc_nz/commit/c174907c23dac5978cabd0671ac195ba24ea889c))
* **c14:** bind releases to verified reproducible artifacts ([e444397](https://github.com/edithatogo/vop_poc_nz/commit/e44439784442a03d6d3777ae84a0066a62fbed7a))
* **c14:** close assurance policy gaps ([332548c](https://github.com/edithatogo/vop_poc_nz/commit/332548cbb5e4c9b33f424837a0f89367a08a4cf3))
* **c14:** close schema and network assurance gaps ([5dbaf08](https://github.com/edithatogo/vop_poc_nz/commit/5dbaf08a5ac70267cf21ce1114024fc87e89fd06))
* **c14:** harden governance drift evidence ([b95ce86](https://github.com/edithatogo/vop_poc_nz/commit/b95ce862444dcad5c45d3f65cc6e76b982167ede))
* **c14:** narrow validated field descriptors ([35aba0d](https://github.com/edithatogo/vop_poc_nz/commit/35aba0d60508a8e595c28b6f9e83a08ec88158ca))
* **c14:** stage releases privately before publication ([527c1d0](https://github.com/edithatogo/vop_poc_nz/commit/527c1d02e5689affcb72a4492e1c1970954d6b98))
* **c14:** type validated field mappings ([361c388](https://github.com/edithatogo/vop_poc_nz/commit/361c38809964d6be9da30f1469a9847de82a8cdf))
* **c15:** bind source heads and normalize sdists ([620fe36](https://github.com/edithatogo/vop_poc_nz/commit/620fe364002c304338bb033b4f2827f9e4c00bdc))
* **c15:** bind VOP assurance cohorts and artifacts ([47058c4](https://github.com/edithatogo/vop_poc_nz/commit/47058c4a9380733c569beebcfa327e4b256df684))
* **c15:** close assurance evidence gaps ([fbeeef4](https://github.com/edithatogo/vop_poc_nz/commit/fbeeef46fe5640f4a06cdb4a35acb1f4f0129e90))
* **c15:** close assurance review findings ([00e5dec](https://github.com/edithatogo/vop_poc_nz/commit/00e5dec9cff2a385fb551772c5722c9907633aca))
* **c15:** exclude runner artifacts from sdist ([7ec3faa](https://github.com/edithatogo/vop_poc_nz/commit/7ec3faa66d6931fe5fdc96007fcb2c8111a01062))
* **c15:** harden operational assurance contracts ([700852f](https://github.com/edithatogo/vop_poc_nz/commit/700852f61a367a8bf3365e267009f2460e44fc00))
* **c15:** normalize packaged text across platforms ([bc7e60d](https://github.com/edithatogo/vop_poc_nz/commit/bc7e60daa8a7f29f01aa29c05924f94d5522099e))
* **ci:** complete mutation sandbox and schema export ([4835e9d](https://github.com/edithatogo/vop_poc_nz/commit/4835e9df145fb0be84740750b54ae3d840bb682c))
* **ci:** copy mutation support directory ([8a42db3](https://github.com/edithatogo/vop_poc_nz/commit/8a42db3836c7503a43870871f70bbc704ec7f58a))
* **ci:** enforce bounded mutation score ([ecdc518](https://github.com/edithatogo/vop_poc_nz/commit/ecdc518d0b23ea022fd9ee341e162bc5be2765ad))
* **ci:** enforce truthful dual mutation gates ([05c6ad0](https://github.com/edithatogo/vop_poc_nz/commit/05c6ad0bd5cadaf3d2eb4a86ebc88c34109210fa))
* **ci:** fetch provenance history for governance gate ([b2a1459](https://github.com/edithatogo/vop_poc_nz/commit/b2a1459f11ef094487957b950bdd472b6676a3bd))
* **ci:** fetch provenance history in test matrix ([2d1b285](https://github.com/edithatogo/vop_poc_nz/commit/2d1b285e210bb5160d7feb717a43b99f45392ecd))
* **ci:** optimize memory usage and enhance CI pipeline ([e7bf1a6](https://github.com/edithatogo/vop_poc_nz/commit/e7bf1a6934f5be5e9c5075cd8571c1d3b09f1d41))
* **ci:** ratchet mutation debt per target ([490f0a7](https://github.com/edithatogo/vop_poc_nz/commit/490f0a7f9e4ff543d289520bc1bd8656cfc0ec41))
* **ci:** remove python 3.14/3.15 from job matrix ([42f0e39](https://github.com/edithatogo/vop_poc_nz/commit/42f0e390ad757c798cb6f5bd14a2bffdc185453e))
* **ci:** use stable Scalene tracer on Python 3.14 ([94bc4bb](https://github.com/edithatogo/vop_poc_nz/commit/94bc4bbf61bcdc7dff794bfeda4e100a0aea986a))
* correct API doc references to existing modules ([0665538](https://github.com/edithatogo/vop_poc_nz/commit/0665538d49da1d3d1664b0294aeaa6b686896218))
* correct module name in docs and fix flaky test ([422eebb](https://github.com/edithatogo/vop_poc_nz/commit/422eebb9de9f833621409066f53b8bc11d2a3c5b))
* Correct NameError for total_gain in run_dcea ([0195927](https://github.com/edithatogo/vop_poc_nz/commit/0195927fe8ec25e4fb4648d490d2751f474346f6))
* Correctly handle discount_rate in subgroup analysis ([b267d58](https://github.com/edithatogo/vop_poc_nz/commit/b267d5871b417788dd3175b16be2de08f0a31359))
* **deps:** add missing PyYAML and hypothesis dependencies ([cf9f5af](https://github.com/edithatogo/vop_poc_nz/commit/cf9f5afb77677055bb495298ac3977476dad7aed))
* **deps:** restore pydantic, pandera, jinja2, tabulate to runtime dependencies ([6a265d5](https://github.com/edithatogo/vop_poc_nz/commit/6a265d55879ae4610fe0455f0dfcba8a8268861c))
* **docs:** correct relative links in quickstart ([6fb0fab](https://github.com/edithatogo/vop_poc_nz/commit/6fb0fabaf89ad63c53c5112c9a2913d1c77306a3))
* extend deptry ignore rules for optional and transitive dependencies ([aac67bb](https://github.com/edithatogo/vop_poc_nz/commit/aac67bb638f8ae9580f5babc95c69c5ce395c6e3))
* final CI fixes ([5ed5b11](https://github.com/edithatogo/vop_poc_nz/commit/5ed5b1150fea2035aea79ef53b6aba193422a0c6))
* format tests and exclude .tox from deptry ([9d8bf6b](https://github.com/edithatogo/vop_poc_nz/commit/9d8bf6b96023e3d68212bd04428e3f2dc305ef3c))
* **governance:** bind local evidence provenance ([8507b50](https://github.com/edithatogo/vop_poc_nz/commit/8507b5003c8759a82e1f934924bfa02063222046))
* **governance:** fail closed on private issue links ([d4f7447](https://github.com/edithatogo/vop_poc_nz/commit/d4f7447cca8fcfc1b466c25986b596de9c4ac9dc))
* Handle list type for intervention_specific_costs in _calculate_friction_cost ([8cbd490](https://github.com/edithatogo/vop_poc_nz/commit/8cbd4901db99da6864780af5d02de597e4f7049f))
* Handle missing discount_rate gracefully in reporting.py and remove debug prints ([27b76d9](https://github.com/edithatogo/vop_poc_nz/commit/27b76d967bf17cab97a7261fe3a036a42bc5bfda))
* **harness:** make modern mutation sandbox self-contained ([da30c45](https://github.com/edithatogo/vop_poc_nz/commit/da30c455a81a60b22306a4d0c4e7514c4f945efc))
* ignore RUF043 regex metacharacter warnings in tests ([50fa721](https://github.com/edithatogo/vop_poc_nz/commit/50fa7212901d9f6ea448e2fcbb6241867ef2bd7d))
* Import copy module in reporting.py ([0a4dd6d](https://github.com/edithatogo/vop_poc_nz/commit/0a4dd6de9b162f4ac7f11800856676924950a13e))
* Import plot_decision_tree in main_analysis.py ([0985dbe](https://github.com/edithatogo/vop_poc_nz/commit/0985dbe867178d13c29543201bd390832e603b9d))
* **imports:** make legacy DSA harness collection-safe ([6913ecd](https://github.com/edithatogo/vop_poc_nz/commit/6913ecde9279c1edf2e8a9b14844c862d8d1c56b))
* **interchange:** bind Arrow identity and scientific units ([ec72bd6](https://github.com/edithatogo/vop_poc_nz/commit/ec72bd6e01d1731904afb2b4ffb4172b733b43c4))
* keep OTLP contract syntax portable ([#49](https://github.com/edithatogo/vop_poc_nz/issues/49)) ([a144dff](https://github.com/edithatogo/vop_poc_nz/commit/a144dff464f06073dff03ca430ddadb5a613c340))
* Make EVPI calculation robust against floating point noise and scalar ambiguity ([1ae27c5](https://github.com/edithatogo/vop_poc_nz/commit/1ae27c588fe303550cd1a319d612d9b116c9feee))
* Make graphviz import optional in visualizations.py to resolve CI ImportError ([9983070](https://github.com/edithatogo/vop_poc_nz/commit/99830700725f1f99e2166aa7ceafac92913d12e0))
* normalize GitHub issue newline transport ([#46](https://github.com/edithatogo/vop_poc_nz/issues/46)) ([1fc10b5](https://github.com/edithatogo/vop_poc_nz/commit/1fc10b57af0c33925e67bb19a74b1f0131cf6247))
* Pass discount_rate as argument to run_cea ([049f6fd](https://github.com/edithatogo/vop_poc_nz/commit/049f6fdc4874f3e6150820b57e8ccec5a70b8f0c))
* **plots:** use memory-safe publication DPI ([79ce523](https://github.com/edithatogo/vop_poc_nz/commit/79ce523520ab7654805ac7b5eef56972ad1d8623))
* Preserve discount_rate in subgroup analysis ([bb6fdd7](https://github.com/edithatogo/vop_poc_nz/commit/bb6fdd7623d3eaa31a482374febef9f256792f11))
* Preserve discount_rate in subgroup analysis by explicitly setting it ([1e7a7a0](https://github.com/edithatogo/vop_poc_nz/commit/1e7a7a0dc424c076178713181fef51740e2de5e0))
* **profile:** adopt Scalene 2.3 command model ([591246a](https://github.com/edithatogo/vop_poc_nz/commit/591246a45bb6f01a52f13d9f4647395da382d443))
* PSA visualization and profiling enhancements ([b5c838a](https://github.com/edithatogo/vop_poc_nz/commit/b5c838a38abccca28d940d6b5e7adbf79b173fbd))
* **quality:** modernize mutation and security gates ([fb0cc1f](https://github.com/edithatogo/vop_poc_nz/commit/fb0cc1f4e2d1536aa2ad324636bb09951df4718d))
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
* **c14:** close assurance frontier track ([58760de](https://github.com/edithatogo/vop_poc_nz/commit/58760deb83ab9e2f7a8b9f2ad118b6d8b7eeb98d))
* **c14:** trace requirements and assurance design ([52a586f](https://github.com/edithatogo/vop_poc_nz/commit/52a586f8dca2c0f8b5a741e6fbcd578ed5beccbf))
* **conductor:** add requirements design and roadmap evidence ([1a0a02e](https://github.com/edithatogo/vop_poc_nz/commit/1a0a02e02a17664ebf12dcb4c89deedebebb6a96))
* **conductor:** close C13 implementation track ([f74a83c](https://github.com/edithatogo/vop_poc_nz/commit/f74a83c5d5a6b76e96179303e8efd1482082a8bd))
* **conductor:** close repository-owned C15 work ([c0783d3](https://github.com/edithatogo/vop_poc_nz/commit/c0783d3b2f623c44ccee7b933ef270294cb54f2b))
* **conductor:** define abstraction excellence requirements ([492c951](https://github.com/edithatogo/vop_poc_nz/commit/492c951b3d20d21f30c7e64c4e077640d6bfc9df))
* **conductor:** record C13 hardening evidence ([4a71370](https://github.com/edithatogo/vop_poc_nz/commit/4a713704c03a3ee3666959c47472b3cb7f55c8b6))
* Integrate new Tutorials and Examples pages into the MkDocs site navigation. ([3590627](https://github.com/edithatogo/vop_poc_nz/commit/359062765f417f732c7174d1973cd1114abfc280))
* reframe documentation around Value of Perspective (VoP) concept ([a2e250f](https://github.com/edithatogo/vop_poc_nz/commit/a2e250f2743805f25da1702f31594ee30079557f))
* update README badges to point to actions ([0b9ad22](https://github.com/edithatogo/vop_poc_nz/commit/0b9ad22f6433ae99f2cad9c71c88e6af25e5f343))
* Update README with DCEA focus and new features ([41ec283](https://github.com/edithatogo/vop_poc_nz/commit/41ec283ca90c152795d3cd34b722b1a22ec11d06))


### Code Refactoring

* Consolidate cost-effectiveness plane functions ([e70265e](https://github.com/edithatogo/vop_poc_nz/commit/e70265eca4d45dd9fb9b1c55901640e7c092b72b))
* **imports:** consolidate legacy module implementations ([02982e5](https://github.com/edithatogo/vop_poc_nz/commit/02982e5d48430454de26c746066d7d675b029e55))
* **imports:** establish canonical package boundary ([09812ee](https://github.com/edithatogo/vop_poc_nz/commit/09812ee8c8267afaa87d20fb0a4e00e4fc3287bf))
* **validation:** use current Pandera import ([c1dbfd3](https://github.com/edithatogo/vop_poc_nz/commit/c1dbfd34379930da442055b0c84e15e6698a7e40))


### Tests

* add CEA model core coverage tests (11/13 passing) ([eb92b76](https://github.com/edithatogo/vop_poc_nz/commit/eb92b76f48ab6d7a8b1ca2f65c50ba7cf9263ee3))
* add comprehensive coverage tests for DCEA, Sobol, tables, and visualizations ([f011161](https://github.com/edithatogo/vop_poc_nz/commit/f011161e227ca88ae1bc03b377d6b48ac55b1c54))
* Add DSA enhancement test script ([179a591](https://github.com/edithatogo/vop_poc_nz/commit/179a5914acb3b75886cd32ab96ea4336cf3e7fa5))
* add value of information coverage tests (7/10 passing) ([90f6111](https://github.com/edithatogo/vop_poc_nz/commit/90f61116bec326622b0efafa9463a67a090cd2db))
* **c14:** ratchet expanded logging mutation baseline ([efcfec2](https://github.com/edithatogo/vop_poc_nz/commit/efcfec2df9e770e3b94a0bf8d954d4c5cc4aecb8))
* **c14:** retain previous-current migration fixture ([55090d4](https://github.com/edithatogo/vop_poc_nz/commit/55090d4fc6e3e5a80417489c89eece7e2593ea8e))
* **mutation:** add strict C13 invariant lane ([bea1cd3](https://github.com/edithatogo/vop_poc_nz/commit/bea1cd3c5767628d067a80a790128996caceb6dd))
* **mutation:** exercise snapshot JSON boundary ([e671a87](https://github.com/edithatogo/vop_poc_nz/commit/e671a870573d51d922a82f7357941c14f3f362ed))


### CI/CD

* align harness with Python 3.14 ([88657f7](https://github.com/edithatogo/vop_poc_nz/commit/88657f70b470bd5bbb3990400ab6c1f9b122d310))
* bound free-threaded wheel probe ([6ef2629](https://github.com/edithatogo/vop_poc_nz/commit/6ef26291d9e81dba79e2903d36a6b76ff4f9e6a3))
* bump actions/checkout from 4 to 6 ([cfd63bc](https://github.com/edithatogo/vop_poc_nz/commit/cfd63bce5f6103c0956547299ca677b46d6ba061))
* bump actions/checkout from 4 to 6 ([c074374](https://github.com/edithatogo/vop_poc_nz/commit/c0743744b54e97e08afaff151b219aeb067f4f75))
* bump actions/setup-python from 5 to 6 ([18303c8](https://github.com/edithatogo/vop_poc_nz/commit/18303c85f652d59b89a01f6e225b12f6b022973f))
* bump actions/setup-python from 5 to 6 ([2c1ba77](https://github.com/edithatogo/vop_poc_nz/commit/2c1ba77a15fead0eb76db91051d9248aedaa4d60))
* bump codecov/codecov-action from 4 to 5 ([945560a](https://github.com/edithatogo/vop_poc_nz/commit/945560a9a5a045cf198cc0474fe741d8b3a12ea1))
* bump codecov/codecov-action from 4 to 5 ([6d831de](https://github.com/edithatogo/vop_poc_nz/commit/6d831de009dee301d913985e9654fd2350455f62))
* **c13:** retain security and experimental evidence ([ea418c7](https://github.com/edithatogo/vop_poc_nz/commit/ea418c7cd41eb304449d689ac2c7bec8b2f2b24c))
* fix pre-commit and tox lint issues ([11ecbe5](https://github.com/edithatogo/vop_poc_nz/commit/11ecbe56d5ca348ceb1633300057a5524e176f9e))
* **governance:** enforce typed contract harness ([b250360](https://github.com/edithatogo/vop_poc_nz/commit/b250360234352f233c0b73ea1b0f8cdf04b0d1d7))
* pin setup-uv action ([1283605](https://github.com/edithatogo/vop_poc_nz/commit/128360584e232c0a8e685a0acccaf159f6dcd04c))
* pin uv across every workflow lane ([#44](https://github.com/edithatogo/vop_poc_nz/issues/44)) ([34eba3d](https://github.com/edithatogo/vop_poc_nz/commit/34eba3d05d54c7d0ed72ae2ca3f59b4384700f87))
* promote reviewed mutation and safe release staging ([422a05f](https://github.com/edithatogo/vop_poc_nz/commit/422a05fdf6459ce92849e976fdf7f1324618eead))
* report free-threaded readiness ([fbd27f3](https://github.com/edithatogo/vop_poc_nz/commit/fbd27f3702c9c921ddef8946e6d3a7c9e8d3e2b6))
* scope lint to maintained Arrow frontier ([8c05e0a](https://github.com/edithatogo/vop_poc_nz/commit/8c05e0a977b555dfeab8fb26f5e9d05baa49fd55))
* support Python 3.12 through 3.14 ([#47](https://github.com/edithatogo/vop_poc_nz/issues/47)) ([8d2345a](https://github.com/edithatogo/vop_poc_nz/commit/8d2345ac916a1dbd61a21f67f08b34532076f02b))

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
