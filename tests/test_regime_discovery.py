from __future__ import annotations

import numpy as np

from vop_poc_nz.regime_discovery import discover_regimes, kmeans


def test_kmeans_discovers_two_separated_regimes() -> None:
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.2, 9.9]])
    result = kmeans(matrix, k=2)
    assert sorted(np.bincount(result.labels).tolist()) == [2, 2]
    assert result.inertia >= 0


def test_discover_regimes_summarises_records() -> None:
    records = [
        {"evop": 0, "budget": 10, "decision": "stable"},
        {"evop": 1, "budget": 11, "decision": "stable"},
        {"evop": 50, "budget": 100, "decision": "discordant"},
        {"evop": 55, "budget": 98, "decision": "discordant"},
    ]
    result = discover_regimes(records, feature_columns=["evop", "budget"], k=2, label_column="decision")
    assert result.feature_names == ("evop", "budget")
    assert sum(summary["n"] for summary in result.regime_summaries) == 4
    assert any("label_counts" in summary for summary in result.regime_summaries)
