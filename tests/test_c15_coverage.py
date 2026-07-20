from __future__ import annotations

from copy import deepcopy

from scripts.check_c15_coverage import evaluate_coverage


def _fixture() -> tuple[dict[str, object], dict[str, object], dict[str, set[int]]]:
    coverage: dict[str, object] = {
        "totals": {"percent_covered": 96.0},
        "files": {
            "src/vop_poc_nz/critical.py": {
                "summary": {"percent_covered": 100.0},
                "executed_lines": [4, 5],
                "missing_lines": [],
                "executed_branches": [[4, 5], [4, 8]],
                "missing_branches": [],
            }
        },
    }
    policy: dict[str, object] = {
        "aggregate_percent": 95.0,
        "critical_modules": {"src/vop_poc_nz/critical.py": 100.0},
        "changed_line_percent": 95.0,
        "changed_branch_percent": 100.0,
    }
    return coverage, policy, {"src/vop_poc_nz/critical.py": {4, 5}}


def test_coverage_policy_enforces_aggregate_critical_changed_and_branch() -> None:
    coverage, policy, changed = _fixture()
    report = evaluate_coverage(coverage, policy, changed)
    assert report["passed"] is True
    assert report["changed"]["branches"] == 2


def test_uncovered_branch_or_unmeasured_file_fails_closed() -> None:
    coverage, policy, changed = _fixture()
    details = coverage["files"]["src/vop_poc_nz/critical.py"]
    details["executed_branches"] = [[4, 5]]
    details["missing_branches"] = [[4, 8]]
    assert evaluate_coverage(coverage, policy, changed)["passed"] is False
    coverage, policy, changed = _fixture()
    changed["src/vop_poc_nz/new.py"] = {1}
    assert evaluate_coverage(coverage, policy, changed)["passed"] is False


def test_low_aggregate_or_missing_critical_module_fails_closed() -> None:
    coverage, policy, changed = _fixture()
    low = deepcopy(coverage)
    low["totals"]["percent_covered"] = 94.99
    assert evaluate_coverage(low, policy, changed)["passed"] is False
    policy["critical_modules"] = {"src/vop_poc_nz/absent.py": 90.0}
    assert evaluate_coverage(coverage, policy, changed)["passed"] is False
