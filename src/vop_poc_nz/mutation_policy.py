"""Fail-closed mutation-score policy for Mutmut CI/CD statistics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class MutationScore:
    """Validated Mutmut counts and their enforceable score."""

    killed: int
    survived: int
    no_tests: int
    suspicious: int
    timeout: int
    segfault: int
    skipped: int
    interrupted: int
    total: int

    @property
    def eligible(self) -> int:
        """Return every non-skipped mutant, including omitted status buckets."""
        return self.total - self.skipped

    @property
    def percent(self) -> float:
        """Return the killed percentage over eligible mutants."""
        return 100.0 * self.killed / self.eligible if self.eligible else 0.0

    def report(
        self, threshold: float, *, baseline: MutationScore | None = None
    ) -> dict[str, Any]:
        """Return a JSON-safe threshold report."""
        non_decreasing = baseline is None or (
            self.eligible > 0
            and baseline.eligible > 0
            and self.killed * baseline.eligible >= baseline.killed * self.eligible
        )
        passed = all(
            (
                self.interrupted == 0,
                self.eligible > 0,
                self.percent >= threshold,
                non_decreasing,
            )
        )
        report: dict[str, Any] = {
            **asdict(self),
            "eligible": self.eligible,
            "score_percent": round(self.percent, 3),
            "threshold_percent": threshold,
            "non_decreasing": non_decreasing,
            "passed": passed,
        }
        if baseline is not None:
            report.update(
                baseline_killed=baseline.killed,
                baseline_eligible=baseline.eligible,
                baseline_score_percent=round(baseline.percent, 3),
            )
        return report


_FIELDS = {
    "killed": "killed",
    "survived": "survived",
    "no_tests": "no_tests",
    "suspicious": "suspicious",
    "timeout": "timeout",
    "segfault": "segfault",
    "skipped": "skipped",
    "interrupted": "check_was_interrupted_by_user",
    "total": "total",
}


def mutation_score_from_mapping(raw: Mapping[str, object]) -> MutationScore:
    """Validate Mutmut 3.6 ``export-cicd-stats`` JSON."""
    values: dict[str, int] = {}
    for field, source in _FIELDS.items():
        value = raw.get(source)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"mutation statistic {source} must be a non-negative integer"
            )
        values[field] = value
    score = MutationScore(**values)
    accounted = (
        score.killed
        + score.survived
        + score.no_tests
        + score.suspicious
        + score.timeout
        + score.segfault
        + score.skipped
        + score.interrupted
    )
    if score.total < accounted:
        raise ValueError("mutation total is smaller than its reported status counts")
    return score


_STATUS_FIELD_BY_EXIT_CODE: dict[int | None, str | None] = {
    None: None,
    0: "survived",
    1: "killed",
    2: "interrupted",
    3: "killed",
    5: "no_tests",
    24: "timeout",
    33: "no_tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    37: None,
    152: "timeout",
    255: "timeout",
    -24: "timeout",
    -11: "segfault",
    -9: "segfault",
}


def mutation_score_from_meta(path: Path) -> MutationScore:
    """Read one Mutmut 3.6 per-source metadata file without hiding statuses."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    statuses = raw.get("exit_code_by_key") if isinstance(raw, dict) else None
    if not isinstance(statuses, dict):
        raise ValueError(f"invalid Mutmut cache metadata: {path}")
    counts = {field: 0 for field in _FIELDS if field != "total"}
    for mutant, exit_code in statuses.items():
        if not isinstance(mutant, str):
            raise ValueError(f"invalid mutant name in cache metadata: {path}")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            raise ValueError(f"invalid Mutmut exit code {exit_code!r} for {mutant}")
        if exit_code not in _STATUS_FIELD_BY_EXIT_CODE:
            raise ValueError(f"unknown Mutmut exit code {exit_code!r} for {mutant}")
        field = _STATUS_FIELD_BY_EXIT_CODE[exit_code]
        if field is not None:
            counts[field] += 1
    return MutationScore(total=len(statuses), **counts)


def mutation_target_report(
    score: MutationScore, *, baseline_killed: int, baseline_eligible: int
) -> dict[str, Any]:
    """Ratchet exact target score and unresolved mutation debt independently."""
    if not 0 <= baseline_killed <= baseline_eligible or baseline_eligible == 0:
        raise ValueError("target baseline counts are inconsistent")
    unresolved = score.eligible - score.killed
    baseline_unresolved = baseline_eligible - baseline_killed
    score_non_decreasing = (
        score.eligible > 0
        and score.killed * baseline_eligible >= baseline_killed * score.eligible
    )
    debt_non_increasing = unresolved <= baseline_unresolved
    return {
        **asdict(score),
        "eligible": score.eligible,
        "score_percent": round(score.percent, 3),
        "baseline_killed": baseline_killed,
        "baseline_eligible": baseline_eligible,
        "baseline_unresolved": baseline_unresolved,
        "unresolved": unresolved,
        "universe_delta": score.eligible - baseline_eligible,
        "score_non_decreasing": score_non_decreasing,
        "debt_non_increasing": debt_non_increasing,
        "passed": (
            score.interrupted == 0
            and score.eligible > 0
            and score_non_decreasing
            and debt_non_increasing
        ),
    }


def validate_threshold(threshold: float) -> float:
    """Require a meaningful percentage threshold."""
    if not 0.0 < threshold <= 100.0:
        raise ValueError("mutation threshold must be greater than 0 and at most 100")
    return threshold


__all__ = [
    "MutationScore",
    "mutation_score_from_mapping",
    "mutation_score_from_meta",
    "mutation_target_report",
    "validate_threshold",
]
