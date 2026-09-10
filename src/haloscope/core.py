"""Paper-faithful latent-subspace membership estimation.

Equation (7) from Du, Xiao, and Li (NeurIPS 2024):

    zeta_i = 1/k * sum_j sigma_j * <f_i - mu, v_j>^2
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    score_centered: bool = True
    score_mode: str = "equation7"
    deterministic_component_sign: bool = False
    factorization_backend: str = "numpy_full"


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

    def fit(self, embeddings: np.ndarray) -> LatentSubspace:
        if self.config.factorization_backend not in {"numpy_full", "sklearn_auto"}:
            raise ValueError(
                "factorization_backend must be numpy_full or sklearn_auto"
            )
        # The released code feeds float32 activation arrays to sklearn PCA.  Keep
        # that dtype for operational parity; the equation-focused implementation
        # retains float64 NumPy SVD for numerical stability.
        dtype = (
            np.float32
            if self.config.factorization_backend == "sklearn_auto"
            else np.float64
        )
        x = _matrix(embeddings).astype(dtype, copy=False)
        if self.config.score_mode not in {"equation7", "official"}:
            raise ValueError("score_mode must be equation7 or official")
        max_components = min(x.shape)
        if not 1 <= self.config.n_components <= max_components:
            raise ValueError(
                f"n_components must be in [1, {max_components}], "
                f"got {self.config.n_components}"
            )
        if self.config.factorization_backend == "sklearn_auto":
            if not self.config.center:
                raise ValueError("sklearn_auto reproduces centered sklearn PCA only")
            try:
                from sklearn.decomposition import PCA
            except ImportError as exc:
                raise RuntimeError(
                    "factorization_backend=sklearn_auto requires scikit-learn"
                ) from exc
            pca = PCA(
                n_components=self.config.n_components,
                whiten=False,
                svd_solver="auto",
                random_state=None,
            ).fit(x)
            self.mean_ = pca.mean_.copy()
            vh = pca.components_.copy()
            singular_values = pca.singular_values_.copy()
        else:
            self.mean_ = x.mean(axis=0) if self.config.center else np.zeros(x.shape[1])
            centered = x - self.mean_
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if (
            self.config.factorization_backend == "numpy_full"
            and self.config.deterministic_component_sign
        ):
            # Match sklearn PCA's svd_flip(..., u_based_decision=False), used
            # by the released implementation. Sign matters because its score
            # averages component projections before taking an absolute value.
            maxima = np.argmax(np.abs(vh), axis=1)
            signs = np.sign(vh[np.arange(vh.shape[0]), maxima])
            signs[signs == 0] = 1
            vh *= signs[:, None]
        k = self.config.n_components
        self.components_ = vh[:k].copy()
        self.singular_values_ = singular_values[:k].copy()
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        x = np.asarray(embeddings, dtype=self.components_.dtype)
        if x.ndim != 2 or x.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"embeddings must have shape [samples, {self.components_.shape[1]}], "
                f"got {x.shape}"
            )
        origin = self.mean_ if self.config.score_centered else 0.0
        return (x - origin) @ self.components_.T

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return zeta; larger values are the paper's hallucination candidates."""
        projected = self.transform(embeddings)
        weights = self.singular_values_ if self.config.weighted else np.ones(projected.shape[1])
        if self.config.score_mode == "official":
            # Released code weights each projected coordinate, averages the
            # coordinates, then takes the magnitude of that scalar.
            return np.abs(np.mean(projected * weights[None, :], axis=1))
        return np.mean(projected**2 * weights[None, :], axis=1)

    def truncated(self, n_components: int) -> LatentSubspace:
        """Reuse a full NumPy SVD for a smaller k without refactorizing."""
        self._require_fitted()
        if not 1 <= n_components <= len(self.singular_values_):
            raise ValueError(
                f"n_components must be in [1, {len(self.singular_values_)}]"
            )
        result = LatentSubspace(replace(self.config, n_components=n_components))
        result.mean_ = self.mean_.copy()
        result.components_ = self.components_[:n_components].copy()
        result.singular_values_ = self.singular_values_[:n_components].copy()
        return result

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
            score_centered=np.array(int(self.config.score_centered)),
            score_mode=np.array(self.config.score_mode),
            deterministic_component_sign=np.array(
                int(self.config.deterministic_component_sign)
            ),
            factorization_backend=np.array(self.config.factorization_backend),
        )

    @classmethod
    def load(cls, path: str | Path) -> LatentSubspace:
        with np.load(path) as state:
            config = SubspaceConfig(
                n_components=int(state["n_components"]),
                weighted=bool(state["weighted"]),
                center=bool(state["center"]),
                score_centered=(
                    bool(state["score_centered"]) if "score_centered" in state else True
                ),
                score_mode=(
                    str(state["score_mode"].item())
                    if "score_mode" in state
                    else "equation7"
                ),
                deterministic_component_sign=(
                    bool(state["deterministic_component_sign"])
                    if "deterministic_component_sign" in state
                    else False
                ),
                factorization_backend=(
                    str(state["factorization_backend"].item())
                    if "factorization_backend" in state
                    else "numpy_full"
                ),
            )
            model = cls(config)
            # Preserve float32 for released-code parity. Upcasting the saved
            # sklearn PCA arrays here changes the raw projection calculation.
            model.mean_ = state["mean"].copy()
            model.components_ = state["components"].copy()
            model.singular_values_ = state["singular_values"].copy()
        return model

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("LatentSubspace must be fitted before use")
