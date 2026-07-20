from __future__ import annotations

from vop_poc_nz.claim_lint import lint_text, format_findings


def test_claim_lint_flags_unbounded_claims() -> None:
    findings = lint_text("This is an empirical demonstration and the first validated model.")
    assert {finding.rule_id for finding in findings} >= {"empirical-demonstration", "policy-grade", "first-novel"}
    assert "empirical" in format_findings(findings)


def test_claim_lint_allows_claims_with_markers() -> None:
    text = "This is an empirically parameterised simulation using evidence_ledger row E1."
    assert lint_text(text) == []
    text_2 = "The intervention is cost-saving under the assumption_ledger threshold scenario."
    assert lint_text(text_2) == []
