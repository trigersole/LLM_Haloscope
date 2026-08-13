"""Paper-faithful latent-subspace membership estimation.

Equation (7) from Du, Xiao, and Li (NeurIPS 2024):

    zeta_i = 1/k * sum_j sigma_j * <f_i - mu, v_j>^2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _matrix(value: np.ndarray, name: str = "embeddings") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, hidden_dim], got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two samples and one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def validate_layerwise(value: np.ndarray, name: str = "embeddings") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [samples, layers, hidden_dim], got {array.shape}")
    if min(array.shape) < 1:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class SubspaceConfig:
    n_components: int = 5
    weighted: bool = True
    center: bool = True


class LatentSubspace:
    """Fit and score the low-dimensional activation subspace used by HaloScope."""

    def __init__(self, config: SubspaceConfig | None = None):
        self.config = config or SubspaceConfig()
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.components_ is not None

    def fit(self, embeddings: np.ndarray) -> "LatentSubspace":
        x = _matrix(embeddings)
        max_components = min(x.shape)
        if not 1 <= self.config.n_components <= max_components:
            raise ValueError(
                f"n_components must be in [1, {max_components}], "
                f"got {self.config.n_components}"
            )
        self.mean_ = x.mean(axis=0) if self.config.center else np.zeros(x.shape[1])
        centered = x - self.mean_
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        k = self.config.n_components
        self.components_ = vh[:k].copy()
        self.singular_values_ = singular_values[:k].copy()
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"embeddings must have shape [samples, {self.components_.shape[1]}], "
                f"got {x.shape}"
            )
        return (x - self.mean_) @ self.components_.T

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return zeta; larger values are the paper's hallucination candidates."""
        projected = self.transform(embeddings)
        weights = self.singular_values_ if self.config.weighted else np.ones(projected.shape[1])
        return np.mean(projected**2 * weights[None, :], axis=1)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean_,
            components=self.components_,
            singular_values=self.singular_values_,
            n_components=np.array(self.config.n_components),
            weighted=np.array(int(self.config.weighted)),
            center=np.array(int(self.config.center)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LatentSubspace":
        with np.load(path) as state:
            config = SubspaceConfig(
                n_components=int(state["n_components"]),
                weighted=bool(state["weighted"]),
                center=bool(state["center"]),
            )
            model = cls(config)
            model.mean_ = state["mean"].astype(np.float64)
            model.components_ = state["components"].astype(np.float64)
            model.singular_values_ = state["singular_values"].astype(np.float64)
        return model

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("LatentSubspace must be fitted before use")

