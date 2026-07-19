"""Perspective regime discovery utilities.

These helpers support the factorial/global-sensitivity + clustering workflow:
turn high-dimensional simulation outputs into a small number of interpretable
regimes. The implementation is dependency-light and deterministic so it can be
used as a reference fixture before a richer scikit-learn/JAX backend is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class RegimeDiscoveryError(ValueError):
    """Invalid regime-discovery input."""


@dataclass(frozen=True)
class RegimeDiscoveryResult:
    """Cluster assignments and summaries for perspective regimes."""

    labels: NDArray[np.int64]
    centroids: NDArray[np.float64]
    feature_names: tuple[str, ...]
    inertia: float
    regime_summaries: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels.tolist(),
            "centroids": self.centroids.tolist(),
            "feature_names": list(self.feature_names),
            "inertia": self.inertia,
            "regime_summaries": [dict(summary) for summary in self.regime_summaries],
        }


def standardize_features(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return z-scored features plus means and scales."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise RegimeDiscoveryError("Feature matrix must be two-dimensional.")
    if matrix.shape[0] < 2:
        raise RegimeDiscoveryError("At least two rows are required for clustering.")
    if not np.all(np.isfinite(matrix)):
        raise RegimeDiscoveryError("Feature matrix must be finite.")
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return (matrix - means) / scales, means, scales


def _initial_centroids(matrix: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Deterministic farthest-point initialisation."""
    centroids = [matrix[0]]
    while len(centroids) < k:
        stacked = np.vstack(centroids)
        distances = np.min(np.sum((matrix[:, None, :] - stacked[None, :, :]) ** 2, axis=2), axis=1)
        centroids.append(matrix[int(np.argmax(distances))])
    return np.vstack(centroids)


def kmeans(
    values: NDArray[np.float64],
    *,
    k: int,
    max_iter: int = 100,
    standardize: bool = True,
) -> RegimeDiscoveryResult:
    """Run deterministic k-means clustering.

    This is a reference implementation for small/medium analysis surfaces. For
    production-scale factorial designs, a later conductor track can switch to
    scikit-learn, Polars, or JAX without changing the output contract.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if standardize:
        matrix, _, _ = standardize_features(matrix)
    if k < 1 or k > matrix.shape[0]:
        raise RegimeDiscoveryError("k must be between 1 and the number of rows.")
    centroids = _initial_centroids(matrix, k)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances = np.sum((matrix[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = matrix[new_labels == cluster]
            if members.size:
                new_centroids[cluster] = np.mean(members, axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    inertia = float(np.sum((matrix - centroids[labels]) ** 2))
    return RegimeDiscoveryResult(
        labels=labels,
        centroids=centroids,
        feature_names=tuple(f"feature_{idx}" for idx in range(matrix.shape[1])),
        inertia=inertia,
    )


def discover_regimes(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    k: int,
    label_column: str | None = None,
) -> RegimeDiscoveryResult:
    """Cluster long-form factorial or GSA records by selected features."""
    if not records:
        raise RegimeDiscoveryError("records must not be empty.")
    matrix = np.asarray(
        [[float(record[column]) for column in feature_columns] for record in records],
        dtype=np.float64,
    )
    result = kmeans(matrix, k=k, standardize=True)
    summaries: list[Mapping[str, Any]] = []
    for cluster in range(k):
        members = [record for record, label in zip(records, result.labels, strict=True) if int(label) == cluster]
        summary: dict[str, Any] = {
            "regime_id": int(cluster),
            "n": len(members),
        }
        if label_column is not None:
            counts: dict[str, int] = {}
            for member in members:
                key = str(member[label_column])
                counts[key] = counts.get(key, 0) + 1
            summary["label_counts"] = counts
        for column in feature_columns:
            values = np.asarray([float(member[column]) for member in members], dtype=np.float64)
            if values.size:
                summary[f"{column}_mean"] = float(np.mean(values))
        summaries.append(summary)
    return RegimeDiscoveryResult(
        labels=result.labels,
        centroids=result.centroids,
        feature_names=tuple(feature_columns),
        inertia=result.inertia,
        regime_summaries=tuple(summaries),
    )
