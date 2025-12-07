# Comprehensive Implementation Plan: vop_poc_nz JSS Submission

## Goal
Expand the `vop_poc_nz` manuscript and software to meet JSS standards (~20 pages) and address expert feedback from Health Economics, Statistical Software, Data Science, Bioethics, Policy, and UQ perspectives.

## User Review Required
> [!IMPORTANT]
> This plan represents a major expansion of the project scope. Please review the "Software Updates" section to ensure the proposed normative and diagnostic features align with your vision.

## 1. Manuscript Expansion (Target: ~8000 words)

### 1.1 Methodology Section (Addressing Health Econ & UQ)
*   **Define DCEA**: Write LaTeX equations for Equity-weighted Net Benefit ($NMB_{eq}$).
*   **Define VoP**: Formalize the Value of Perspective equation:
    $$ VoP = \sum_{s} p(s) \cdot \max_a E[NB(a, s)] - \max_a \sum_{s} p(s) E[NB(a, s)] $$
    (or the specific formulation used in the code).
*   **Monte Carlo Integration**: Describe the simulation algorithm and how nested uncertainty is handled (or approximated).

### 1.2 Normative Framework Section (Addressing Bioethics)
*   **Ethical Foundations**: Explicitly discuss the choice of the Atkinson index and the meaning of inequality aversion ($\epsilon$).
*   **Value Judgments**: Acknowledge that "equity weights" are moral parameters, not physical constants.

### 1.3 Software Architecture Section (Addressing Software Eng)
*   **Design Patterns**: Document the modular structure (Core Model vs. Pipeline vs. Reporting).
*   **Comparison**: Explicitly position `vop_poc_nz` against `heemod` (R) and `dampack` (R), highlighting it as a **unique Python-native contribution** that introduces VoP, rather than a direct clone.

### 1.4 Case Studies & Results (Addressing Policy & Data Science)
*   **Model Description**: Briefly describe the HPV, Smoking, etc., models (states, cycle length).
*   **Results**: Insert the generated figures/tables directly into the LaTeX manuscript:
    *   Comparative ICER Table.
    *   Discordance Plot (VoP).
    *   CE Plane (Societal vs. Health System).
    *   Policy Brief Example.

## 2. Supplementary Materials

### 2.1 Supplement A: Mathematical Proofs & Diagnostics
*   **Convergence**: Show plots demonstrating that the number of PSA samples is sufficient for stable VoP estimates.
*   **Validation**: Cross-validation against a simple Excel/R model if possible (or unit test evidence).

### 2.2 Supplement B: Guide to Equity Weights
*   **User Guide**: Explain how to choose $\epsilon$ (0.5 vs 1.0 vs 2.0) and what it means for policy.

### 2.3 Supplement C: Data Dictionary
*   **Parameters**: Full documentation of `parameters.yaml` structure and sources.

## 3. Software Updates

### 3.1 Normative Transparency
#### [MODIFY] `src/vop_poc_nz/pipeline/reporting.py`
*   **Feature**: Add a "Normative Assumptions" section to the generated reports.
*   **Content**: Explicitly state the inequality aversion parameter ($\epsilon$) and equity weights used in the analysis.

### 3.2 Convergence Diagnostics
#### [NEW] `src/vop_poc_nz/diagnostics.py`
*   **Feature**: Implement a function to check PSA stability (e.g., running standard error of the mean).
*   **Output**: Generate a convergence plot/report.

### 3.3 Reporting Enhancements
#### [MODIFY] `src/vop_poc_nz/pipeline/reporting.py`
*   **Modular Reporting**: Ensure the `combined_report.md` includes *all* generated metrics and figures.
*   **Policy Brief**: Embed the "Decision Rule" interpretation of VoP (e.g., "If VoP > Cost of Research, collect data").

## 4. Verification
*   **Run Pipeline**: Execute `reproduce_results.py` to generate all new outputs.
*   **Compile Manuscript**: Compile the expanded LaTeX manuscript and check for layout issues.
*   **Review**: Self-review against the "Simulated Peer Review" checklist.
