from __future__ import annotations

from vop_poc_nz.manuscript_audit import audit_manuscript_text, render_manuscript_audit_markdown
from vop_poc_nz.reviewer_matrix import reviewer_response_matrix


def test_manuscript_audit_flags_societal_bonus_and_missing_evop_semantics() -> None:
    text = """
    # Results
    The Value of Perspective identifies a societal bonus and the wrong funding decision.
    """
    issues = audit_manuscript_text(text)
    assert any(issue.code == "ambiguous_or_overclaiming_term" for issue in issues)
    assert any(issue.code == "evop_semantics_underdefined" for issue in issues)


def test_manuscript_audit_markdown_renders() -> None:
    report = render_manuscript_audit_markdown(audit_manuscript_text("No value of perspective here."))
    assert "| Severity | Code |" in report


def test_reviewer_response_matrix_contains_core_tracks() -> None:
    matrix = reviewer_response_matrix()
    assert "track_27_concept_scope_budget.md" in matrix
    assert "Perspective Acceptability Frontier" in matrix
