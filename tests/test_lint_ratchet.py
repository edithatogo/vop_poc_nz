from scripts.lint_ratchet import ratchet_failures


def test_lint_ratchet_rejects_new_or_increased_rules() -> None:
    current = {"E501": 2, "F401": 1, "SIM101": 1}
    maximum = {"E501": 2, "F401": 0}
    assert ratchet_failures(current, maximum) == [
        "F401: 1 exceeds baseline 0",
        "SIM101: 1 exceeds baseline 0",
    ]
