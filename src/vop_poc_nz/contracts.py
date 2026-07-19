"""Lightweight validators for case contracts and evidence ledgers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - import branch depends on optional dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


class ContractError(ValueError):
    """Raised when a case contract or evidence ledger is invalid."""


REQUIRED_CASE_FIELDS = {
    "case_id",
    "case_type",
    "model_family",
    "decision_strategies",
    "perspectives",
    "cost_components",
    "source_grade",
    "validation_status",
}

REQUIRED_LEDGER_COLUMNS = {
    "parameter_id",
    "case_id",
    "value",
    "unit",
    "source_citation",
    "derivation_formula",
    "included_perspectives",
    "included_cost_component",
    "uncertainty_rationale",
}


@dataclass(frozen=True)
class CaseContract:
    """A validated case-study contract."""

    data: Mapping[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.data["case_id"])

    @property
    def case_type(self) -> str:
        return str(self.data["case_type"])

    @property
    def perspectives(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.data["perspectives"])

    @property
    def decision_strategies(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.data["decision_strategies"])


def validate_case_contract(data: Mapping[str, Any]) -> CaseContract:
    """Validate minimal case-contract fields and return a typed wrapper."""
    missing = sorted(REQUIRED_CASE_FIELDS - set(data))
    if missing:
        raise ContractError(f"Case contract is missing required fields: {missing}")
    if not str(data["case_id"]).strip():
        raise ContractError("case_id must be non-empty.")
    if data["case_type"] not in {"policy_grade", "empirical_tutorial", "synthetic_fixture"}:
        raise ContractError(
            "case_type must be one of policy_grade, empirical_tutorial, synthetic_fixture."
        )
    for field in ("decision_strategies", "perspectives", "cost_components"):
        value = data[field]
        if not isinstance(value, list) or not value:
            raise ContractError(f"{field} must be a non-empty list.")
        if len(set(map(str, value))) != len(value):
            raise ContractError(f"{field} entries must be unique.")
    if not isinstance(data["source_grade"], Mapping):
        raise ContractError("source_grade must be a mapping.")
    if not isinstance(data["validation_status"], Mapping):
        raise ContractError("validation_status must be a mapping.")
    return CaseContract(data=dict(data))


def load_case_contract(path: str | Path) -> CaseContract:
    """Load and validate a YAML case contract."""
    if yaml is None:
        raise ContractError("PyYAML is required to load YAML case contracts.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ContractError("Case contract YAML must contain a mapping at the top level.")
    return validate_case_contract(data)


def validate_evidence_ledger_rows(rows: list[Mapping[str, Any]]) -> None:
    """Validate evidence-ledger rows already loaded as dictionaries."""
    if not rows:
        raise ContractError("Evidence ledger must contain at least one row.")
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_LEDGER_COLUMNS - set(row))
        if missing:
            raise ContractError(f"Row {row_number} missing required columns: {missing}")
        parameter_id = str(row["parameter_id"]).strip()
        case_id = str(row["case_id"]).strip()
        if not parameter_id or not case_id:
            raise ContractError(f"Row {row_number} requires non-empty parameter_id and case_id.")
        key = (case_id, parameter_id)
        if key in seen:
            raise ContractError(f"Duplicate parameter_id within case: {key}")
        seen.add(key)
        try:
            float(row["value"])
        except Exception as exc:
            raise ContractError(f"Row {row_number} has non-numeric value.") from exc
        if not str(row["source_citation"]).strip():
            raise ContractError(f"Row {row_number} requires a source_citation.")
        if not str(row["uncertainty_rationale"]).strip():
            raise ContractError(f"Row {row_number} requires an uncertainty_rationale.")


def validate_evidence_ledger_csv(path: str | Path) -> list[dict[str, str]]:
    """Load and validate a CSV evidence ledger."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_evidence_ledger_rows(rows)
    return rows
