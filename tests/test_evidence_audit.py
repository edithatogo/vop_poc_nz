from __future__ import annotations

from vop_poc_nz.evidence_audit import audit_evidence_ledger_rows, evidence_issue_summary


def test_evidence_audit_flags_cost_without_price_context() -> None:
    rows = [
        {
            "parameter_id": "cost_insulation",
            "case_id": "housing",
            "value": "1200",
            "unit": "NZD cost",
            "source_citation": "Example Study 2010",
            "derivation_formula": "reported",
            "included_perspectives": "societal",
            "cost_component": "energy",
            "distribution": "gamma",
            "price_year": "",
            "currency": "",
        }
    ]
    issues = audit_evidence_ledger_rows(rows, current_year=2026)
    summary = evidence_issue_summary(issues)
    assert summary["warning"] >= 2
    assert any(issue.code == "cost_without_price_context" for issue in issues)


def test_evidence_audit_accepts_structured_societal_component() -> None:
    rows = [
        {
            "parameter_id": "productivity_gain",
            "case_id": "smoking",
            "value": "4500",
            "unit": "NZD",
            "source_citation": "Example Study 2024",
            "derivation_formula": "human capital method",
            "included_perspectives": "societal",
            "cost_component": "productivity",
            "distribution": "lognormal",
            "price_year": "2024",
            "currency": "NZD",
        }
    ]
    issues = audit_evidence_ledger_rows(rows, current_year=2026)
    assert not any(issue.code == "unclassified_societal_component" for issue in issues)
    assert not any(issue.severity == "error" for issue in issues)
