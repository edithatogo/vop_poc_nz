"""Dependency-light internal validation helpers for economic models.

These checks do not establish external validity. They provide auditable evidence
for transition matrices, cohort traces, discounting inputs, and PSA samples so
case-specific clinical and epidemiological validation can build on a consistent
base.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationFinding:
    severity: Severity
    code: str
    message: str
    location: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass(frozen=True)
class ValidationReport:
    check: str
    findings: tuple[ValidationFinding, ...]
    metadata: dict[str, Any]

    @property
    def valid(self) -> bool:
        return not any(finding.severity is Severity.ERROR for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "check": self.check,
            "valid": self.valid,
            "metadata": self.metadata,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def _as_float_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def validate_transition_matrix(
    matrix: ArrayLike,
    *,
    atol: float = 1e-10,
    absorbing_state: int | None = None,
) -> ValidationReport:
    """Validate one transition matrix or a draw × state × state stack."""
    array = _as_float_array(matrix, name="matrix")
    findings: list[ValidationFinding] = []
    if array.ndim == 2:
        stack = array[np.newaxis, :, :]
    elif array.ndim == 3:
        stack = array
    else:
        return ValidationReport(
            "transition_matrix",
            (ValidationFinding(Severity.ERROR, "matrix_dimension", "matrix must be 2D or 3D"),),
            {"shape": list(array.shape)},
        )
    if stack.shape[1] != stack.shape[2]:
        findings.append(ValidationFinding(Severity.ERROR, "matrix_not_square", "transition matrices must be square"))
    if not np.all(np.isfinite(stack)):
        findings.append(ValidationFinding(Severity.ERROR, "matrix_non_finite", "transition probabilities must be finite"))
    negative = np.argwhere(stack < -atol)
    if negative.size:
        findings.append(ValidationFinding(Severity.ERROR, "matrix_negative", f"{len(negative)} transition probabilities are negative"))
    above_one = np.argwhere(stack > 1.0 + atol)
    if above_one.size:
        findings.append(ValidationFinding(Severity.ERROR, "matrix_above_one", f"{len(above_one)} transition probabilities exceed one"))
    row_sums = np.sum(stack, axis=2)
    bad_rows = np.argwhere(~np.isclose(row_sums, 1.0, rtol=0.0, atol=atol))
    if bad_rows.size:
        findings.append(ValidationFinding(Severity.ERROR, "matrix_row_sum", f"{len(bad_rows)} rows do not sum to one"))
    if absorbing_state is not None:
        if absorbing_state < 0 or absorbing_state >= stack.shape[1]:
            findings.append(ValidationFinding(Severity.ERROR, "absorbing_state_index", "absorbing_state is outside the state range"))
        else:
            expected = np.zeros(stack.shape[2], dtype=np.float64)
            expected[absorbing_state] = 1.0
            bad = np.argwhere(~np.all(np.isclose(stack[:, absorbing_state, :], expected, rtol=0.0, atol=atol), axis=1))
            if bad.size:
                findings.append(ValidationFinding(Severity.ERROR, "absorbing_state_not_absorbing", f"absorbing-state row is invalid in {len(bad)} draws"))
    if not findings:
        findings.append(ValidationFinding(Severity.INFO, "matrix_valid", "transition matrix checks passed"))
    return ValidationReport(
        "transition_matrix",
        tuple(findings),
        {"shape": list(array.shape), "atol": atol, "absorbing_state": absorbing_state},
    )


def validate_state_trace(
    trace: ArrayLike,
    *,
    atol: float = 1e-10,
    absorbing_state: int | None = None,
) -> ValidationReport:
    """Validate a cycle × state cohort trace."""
    array = _as_float_array(trace, name="trace")
    findings: list[ValidationFinding] = []
    if array.ndim != 2:
        return ValidationReport(
            "state_trace",
            (ValidationFinding(Severity.ERROR, "trace_dimension", "trace must be cycle × state"),),
            {"shape": list(array.shape)},
        )
    if not np.all(np.isfinite(array)):
        findings.append(ValidationFinding(Severity.ERROR, "trace_non_finite", "trace values must be finite"))
    if np.any(array < -atol):
        findings.append(ValidationFinding(Severity.ERROR, "trace_negative", "trace contains negative state occupancy"))
    row_sums = np.sum(array, axis=1)
    bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=0.0, atol=atol))
    if bad_rows.size:
        findings.append(ValidationFinding(Severity.ERROR, "trace_row_sum", f"{len(bad_rows)} cycles do not sum to one"))
    if absorbing_state is not None:
        if absorbing_state < 0 or absorbing_state >= array.shape[1]:
            findings.append(ValidationFinding(Severity.ERROR, "absorbing_state_index", "absorbing_state is outside the state range"))
        else:
            changes = np.diff(array[:, absorbing_state])
            if np.any(changes < -atol):
                findings.append(ValidationFinding(Severity.ERROR, "absorbing_state_decreases", "occupancy of the absorbing state decreases over time"))
    if not findings:
        findings.append(ValidationFinding(Severity.INFO, "trace_valid", "state-trace checks passed"))
    return ValidationReport("state_trace", tuple(findings), {"shape": list(array.shape), "atol": atol, "absorbing_state": absorbing_state})


def validate_discount_rate(rate: float, *, label: str = "discount_rate") -> ValidationReport:
    value = float(rate)
    findings: list[ValidationFinding] = []
    if not np.isfinite(value):
        findings.append(ValidationFinding(Severity.ERROR, "discount_non_finite", f"{label} must be finite"))
    elif value <= -1.0:
        findings.append(ValidationFinding(Severity.ERROR, "discount_invalid", f"{label} must be greater than -1"))
    elif value < 0:
        findings.append(ValidationFinding(Severity.WARNING, "discount_negative", f"{label} is negative and requires explicit justification"))
    elif value > 0.15:
        findings.append(ValidationFinding(Severity.WARNING, "discount_high", f"{label} is unusually high and requires explicit justification"))
    else:
        findings.append(ValidationFinding(Severity.INFO, "discount_valid", f"{label} is numerically valid"))
    return ValidationReport("discount_rate", tuple(findings), {"label": label, "value": value})


def validate_psa_sample(values: ArrayLike, *, min_draws: int = 1000) -> ValidationReport:
    array = _as_float_array(values, name="values")
    findings: list[ValidationFinding] = []
    if array.ndim < 1:
        findings.append(ValidationFinding(Severity.ERROR, "psa_dimension", "PSA values require a draw dimension"))
    if not np.all(np.isfinite(array)):
        findings.append(ValidationFinding(Severity.ERROR, "psa_non_finite", "PSA sample contains non-finite values"))
    draws = int(array.shape[0]) if array.ndim else 0
    if draws < min_draws:
        findings.append(ValidationFinding(Severity.WARNING, "psa_draw_count", f"PSA has {draws} draws; fewer than configured minimum {min_draws}"))
    if array.ndim and np.any(np.nanstd(array, axis=0) == 0):
        findings.append(ValidationFinding(Severity.WARNING, "psa_zero_variance", "At least one PSA output dimension has zero variance"))
    if not findings:
        findings.append(ValidationFinding(Severity.INFO, "psa_valid", "PSA sample checks passed"))
    return ValidationReport("psa_sample", tuple(findings), {"shape": list(array.shape), "min_draws": min_draws})
