# JSS Submission Plan for `vop_poc_nz`

This document outlines the comprehensive plan to prepare the `vop_poc_nz` software for submission to the **Journal of Statistical Software (JSS)**.

## Goal
Submit `vop_poc_nz` to JSS, meeting all requirements for software quality, reproducibility, and manuscript formatting.

## 1. Software Preparation & Cleanup

### 1.1 License Standardization
**Current Status:** Conflicting information. `LICENSE` file is Apache 2.0, while `pyproject.toml` states `MIT`.
**Requirement:** JSS requires a GPL-compatible license.
**Decision:** **Apache 2.0** throughout.
**Action:**
- [ ] Update `pyproject.toml` to specify Apache 2.0.
- [ ] Ensure `LICENSE` file is the correct Apache 2.0 text.
- [ ] Check source files for license headers.

### 1.2 Dependency Management & Packaging
**Current Status:** Mixed configuration.
**Requirement:** Easy installation (PyPI).
**Decision:** Use **Hatch** as the build backend.
**Action:**
- [ ] Configure `pyproject.toml` to use `hatchling` as the build backend.
- [ ] Configure `hatch` environments for testing and docs.
- [ ] Verify package builds and installs correctly.
- [ ] Prepare for PyPI/TestPyPI publication.

### 1.3 Documentation
**Current Status:** `docs/` exists.
**Decision:** Use **MkDocs** (already set up).
**Action:**
- [ ] Ensure `mkdocs.yml` is configured correctly.
- [ ] Verify GitHub Actions workflow for Pages is working.
- [ ] Add a specific "JSS Replication" section to the docs.

## 2. Replication Materials

### 2.1 Standalone Replication Script
**Current Status:** `Snakefile` exists.
**Requirement:** "Single, commented standalone replication script".
**Decision:** Use `Snakefile` as the engine, but provide a Python wrapper `reproduce_results.py` to satisfy the "single script" requirement and handle setup.
**Action:**
- [ ] Create `reproduce_results.py`:
    - Checks/installs dependencies (including Snakemake).
    - Runs the Snakemake workflow.
    - Prints location of outputs.
- [ ] Ensure `Snakefile` is commented and clear.

### 2.2 Data Management
**Current Status:** Data is primarily in `src/parameters.yaml`.
**Action:**
- [ ] Document clearly that `src/parameters.yaml` contains the input data for the case studies.
- [ ] Ensure this file is included in the package distribution (MANIFEST.in or hatch config).

## 3. Manuscript Preparation

### 3.1 JSS LaTeX Template
**Action:**
- [ ] Initialize `manuscript/jss_submission.tex` using the saved JSS style guide and templates.
- [ ] Ensure all LaTeX dependencies are documented.

### 3.2 Content Development & Comparison
**Key Contribution:** `vop_poc_nz` is the first software to **operationalize the quantification of perspective uncertainty** (Value of Perspective).
**Comparison Strategy:**
- **dampack** (R): Focuses on PSA, CEACs, EVPI, but perspective is just an input choice, not a quantified uncertainty.
- **heemod** (R): Markov modeling, allows different cost/utility inputs, but no native "Value of Perspective" analysis.
- **BCEA** (R): Bayesian post-processing, handles PSA/EVPI well, but again, perspective is a fixed assumption per run.
**Action:**
- [ ] Draft the "Software Comparison" section highlighting this unique feature.
- [ ] Contrast the "framework" approach of `vop_poc_nz` (end-to-end DCEA + VOI + Perspective) with the "tool" approach of others.

## 4. Submission Checklist

- [ ] **Code**: Apache 2.0, Hatch build, Linted, Tested.
- [ ] **Package**: On PyPI (or ready to be).
- [ ] **Replication**: `reproduce_results.py` works on a fresh machine.
- [ ] **Manuscript**: JSS style PDF, referencing the software and replication script.
- [ ] **Web**: GitHub Pages up to date.

## Immediate Next Steps
1.  **Fix License**: Update `pyproject.toml` and `LICENSE`.
2.  **Migrate to Hatch**: Reconfigure `pyproject.toml`.
3.  **Verify Docs**: Check MkDocs build.
