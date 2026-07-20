from __future__ import annotations

import pytest

from vop_poc_nz.contracts import (
    ContractError,
    validate_case_contract,
    validate_evidence_ledger_rows,
)


def test_case_contract_validation_accepts_minimal_valid_contract() -> None:
    contract = validate_case_contract(
        {
            "case_id": "housing_insulation_nz",
            "case_type": "empirical_tutorial",
            "model_family": "markov",
            "decision_strategies": ["standard_care", "insulation"],
            "perspectives": ["health_system", "societal"],
            "cost_components": ["direct_medical", "energy_savings"],
            "source_grade": {"clinical": "empirical_published"},
            "validation_status": {"internal": "pending"},
        }
    )

    assert contract.case_id == "housing_insulation_nz"
    assert contract.perspectives == ("health_system", "societal")


def test_case_contract_rejects_unlabelled_case_type() -> None:
    with pytest.raises(ContractError, match="case_type"):
        validate_case_contract(
            {
                "case_id": "x",
                "case_type": "interesting",
                "model_family": "markov",
                "decision_strategies": ["a"],
                "perspectives": ["p"],
                "cost_components": ["c"],
                "source_grade": {},
                "validation_status": {},
            }
        )


def test_evidence_ledger_requires_source_and_uncertainty_rationale() -> None:
    rows = [
        {
            "parameter_id": "mortality_rr",
            "case_id": "housing_insulation_nz",
            "value": "0.95",
            "unit": "relative_risk",
            "source_citation": "Doe 2024",
            "derivation_formula": "reported estimate",
            "included_perspectives": "health_system;societal",
            "included_cost_component": "clinical_effect",
            "uncertainty_rationale": "published confidence interval",
        }
    ]

    validate_evidence_ledger_rows(rows)


def test_evidence_ledger_rejects_duplicate_parameters_within_case() -> None:
    rows = [
        {
            "parameter_id": "x",
            "case_id": "case",
            "value": "1",
            "unit": "nzd",
            "source_citation": "source",
            "derivation_formula": "formula",
            "included_perspectives": "societal",
            "included_cost_component": "cost",
            "uncertainty_rationale": "reason",
        },
        {
            "parameter_id": "x",
            "case_id": "case",
            "value": "2",
            "unit": "nzd",
            "source_citation": "source",
            "derivation_formula": "formula",
            "included_perspectives": "societal",
            "included_cost_component": "cost",
            "uncertainty_rationale": "reason",
        },
    ]

    with pytest.raises(ContractError, match="Duplicate"):
        validate_evidence_ledger_rows(rows)
