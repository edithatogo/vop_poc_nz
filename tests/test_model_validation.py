from __future__ import annotations

import numpy as np

from vop_poc_nz.model_validation import Severity, validate_discount_rate, validate_psa_sample, validate_state_trace, validate_transition_matrix


def test_transition_and_trace_validation() -> None:
    matrix = np.array([[0.8, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])
    assert validate_transition_matrix(matrix, absorbing_state=2).valid
    trace = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.64, 0.34, 0.02]])
    assert validate_state_trace(trace, absorbing_state=2).valid


def test_invalid_transition_matrix_reports_errors() -> None:
    report = validate_transition_matrix(np.array([[0.8, 0.3], [0.0, 1.0]]))
    assert not report.valid
    assert any(item.severity is Severity.ERROR for item in report.findings)


def test_discount_and_psa_warnings_are_auditable() -> None:
    assert validate_discount_rate(-0.01).valid
    psa = validate_psa_sample(np.ones((10, 2)), min_draws=1000)
    assert psa.valid
    assert any(item.severity is Severity.WARNING for item in psa.findings)
