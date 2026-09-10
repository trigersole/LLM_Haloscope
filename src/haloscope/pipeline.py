"""End-to-end HaloScope selection, pseudo-labeling, training, and inference."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .core import LatentSubspace, SubspaceConfig, validate_layerwise
from .metrics import (
    average_precision,
    balanced_accuracy,
    binary_accuracy,
    brier_score,
    expected_calibration_error,
    fpr_at_tpr,
    roc_auc,
)
from .probe import ProbeConfig, TruthfulnessProbe, build_probe, load_probe


def _default_quantiles() -> tuple[float, ...]:
    return tuple(float(x) for x in np.linspace(0.0, 1.0, 40)[1:-1])


@dataclass(frozen=True)
class SearchConfig:
    k_values: tuple[int, ...] = tuple(range(1, 11))
    threshold_quantiles: tuple[float, ...] = field(default_factory=_default_quantiles)
    layers: tuple[int, ...] | None = None
    probe_layers: tuple[int, ...] | None = None
    weighted: bool = True
    center: bool = True
    score_centered: bool = True
    score_mode: str = "equation7"
    deterministic_component_sign: bool = False
    factorization_backend: str = "numpy_full"
    replay_official_numpy_rng: bool = False
    selection_source: str = "wild"
    quantile_method: str = "linear"
    retrain_selected_probe: bool = False
    # "paper": high zeta is hallucinated; "auto": choose direction using validation labels,
    # reproducing the official code's sign search.
    orientation: str = "paper"
    pseudo_label_mode: str = "threshold"
    tail_fractions: tuple[float, ...] = (0.2, 0.25, 0.3)
    confidence_weighted: bool = False
    probe_repeats: int = 1
    validation_folds: int = 1
    stability_penalty: float = 0.0


@dataclass
class FitSummary:
    subspace_layer: int
    probe_layer: int
    n_components: int
    threshold_quantile: float
    threshold_value: float
    truth_score_sign: int
    validation_auroc: float
    validation_direct_auroc: float
    pseudo_truthful: int
    pseudo_hallucinated: int
    pseudo_ignored: int = 0
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    validation_selection_score: float | None = None
    validation_fold_std: float = 0.0
    probe_repeats: int = 1


class HaloScope:
    """Train a truthfulness detector using unlabeled mixture activations."""

    def __init__(
        self,
        search: SearchConfig | None = None,
        probe_config: ProbeConfig | None = None,
    ):
        self.search = search or SearchConfig()
        self.probe_config = probe_config or ProbeConfig()
        self.subspace: LatentSubspace | None = None
        self.probe: TruthfulnessProbe | None = None
        self.probes: list[TruthfulnessProbe] = []
        self.summary: FitSummary | None = None

    def fit(
        self,
        wild_embeddings: np.ndarray,
        validation_embeddings: np.ndarray,
        validation_truth_labels: np.ndarray,
    ) -> HaloScope:
        wild = validate_layerwise(wild_embeddings, "wild_embeddings")
        validation = validate_layerwise(validation_embeddings, "validation_embeddings")
        labels = np.asarray(validation_truth_labels, dtype=np.int64).reshape(-1)
        if wild.shape[1:] != validation.shape[1:] or len(validation) != len(labels):
            raise ValueError("wild/validation embedding shapes or validation labels do not match")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("validation labels must contain truthful (1) and hallucinated (0) samples")
        if self.search.orientation not in {"paper", "auto"}:
            raise ValueError("orientation must be 'paper' or 'auto'")
        if self.search.selection_source not in {"wild", "validation"}:
            raise ValueError("selection_source must be wild or validation")
        if self.search.quantile_method not in {"linear", "official"}:
            raise ValueError("quantile_method must be linear or official")
        if self.search.pseudo_label_mode not in {"threshold", "confidence_tails"}:
            raise ValueError("pseudo_label_mode must be threshold or confidence_tails")
        if self.search.probe_repeats < 1:
            raise ValueError("probe_repeats must be at least 1")
        if self.search.validation_folds < 1:
            raise ValueError("validation_folds must be at least 1")
        if self.search.stability_penalty < 0:
            raise ValueError("stability_penalty must be non-negative")

        if self.probe_config.backend == "torch_mlp" and not self.probe_config.seed_each_fit:
            import torch

            torch.manual_seed(self.probe_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.probe_config.seed)

        layers = self._indices(self.search.layers, wild.shape[1], "layers")
        selection = validation if self.search.selection_source == "validation" else wild
        best_direct = None
        cached_subspaces = {}
        if self.search.factorization_backend == "numpy_full":
            valid_k = [
                value
                for value in self.search.k_values
                if value <= min(selection.shape[0], selection.shape[2])
            ]
            if valid_k:
                max_k = max(valid_k)
                cached_subspaces = {
                    layer: LatentSubspace(self._subspace_config(max_k)).fit(
                        selection[:, layer, :]
                    )
                    for layer in layers
                }
        # Released code iterates k first, then layer; preserve its tie-breaking.
        for k in self.search.k_values:
            for layer in layers:
                if k > min(selection.shape[0], selection.shape[2]):
                    continue
                candidate = (
                    cached_subspaces[layer].truncated(k)
                    if cached_subspaces
                    else LatentSubspace(self._subspace_config(k)).fit(
                        selection[:, layer, :]
                    )
                )
                zeta = candidate.score(validation[:, layer, :])
                sign, selection_score, pooled_auc, fold_std = self._truth_orientation(
                    zeta, labels
                )
                if best_direct is None or selection_score > best_direct[0]:
                    best_direct = (
                        selection_score,
                        pooled_auc,
                        fold_std,
                        layer,
                        k,
                        sign,
                    )
        if best_direct is None:
            raise ValueError("no valid layer/k candidate; reduce k_values or add wild samples")

        _, direct_auc, _, subspace_layer, k, sign = best_direct
        self.subspace = LatentSubspace(
            self._subspace_config(k)
        ).fit(wild[:, subspace_layer, :])
        wild_truth_score = sign * self.subspace.score(wild[:, subspace_layer, :])
        probe_layers = self._indices(self.search.probe_layers, wild.shape[1], "probe_layers")

        best_probe = None
        candidates = (
            self.search.tail_fractions
            if self.search.pseudo_label_mode == "confidence_tails"
            else self.search.threshold_quantiles
        )
        for quantile in candidates:
            pseudo = self._pseudo_labels(wild_truth_score, quantile)
            selected, pseudo_truth, sample_weight, lower_threshold, upper_threshold = pseudo
            if len(np.unique(pseudo_truth)) != 2:
                continue
            for layer in probe_layers:
                candidate_probes = self._fit_probe_ensemble(
                    wild[selected, layer, :], pseudo_truth, sample_weight
                )
                probabilities = self._ensemble_probabilities(
                    candidate_probes, validation[:, layer, :]
                )
                selection_score, pooled_auc, fold_std = self._validation_score(
                    labels, probabilities
                )
                if best_probe is None or selection_score > best_probe[0]:
                    best_probe = (
                        selection_score,
                        pooled_auc,
                        fold_std,
                        layer,
                        quantile,
                        lower_threshold,
                        upper_threshold,
                        selected.copy(),
                        pseudo_truth.copy(),
                        None if sample_weight is None else sample_weight.copy(),
                        candidate_probes,
                    )
        if best_probe is None:
            raise RuntimeError("probe search produced no valid model")

        (
            selection_score,
            auc,
            fold_std,
            probe_layer,
            quantile,
            lower_threshold,
            upper_threshold,
            selected,
            pseudo_truth,
            sample_weight,
            selected_probes,
        ) = best_probe
        self.probes = (
            self._fit_probe_ensemble(
                wild[selected, probe_layer, :], pseudo_truth, sample_weight
            )
            if self.search.retrain_selected_probe
            else selected_probes
        )
        self.probe = self.probes[0]
        self.summary = FitSummary(
            subspace_layer=subspace_layer,
            probe_layer=probe_layer,
            n_components=k,
            threshold_quantile=float(quantile),
            threshold_value=float(
                upper_threshold if self.search.pseudo_label_mode == "threshold"
                else (lower_threshold + upper_threshold) / 2.0
            ),
            truth_score_sign=sign,
            validation_auroc=float(auc),
            validation_direct_auroc=float(direct_auc),
            pseudo_truthful=int(pseudo_truth.sum()),
            pseudo_hallucinated=int((1 - pseudo_truth).sum()),
            pseudo_ignored=int(len(wild) - len(pseudo_truth)),
            lower_threshold=float(lower_threshold),
            upper_threshold=float(upper_threshold),
            validation_selection_score=float(selection_score),
            validation_fold_std=float(fold_std),
            probe_repeats=len(self.probes),
        )
        return self

    def predict_truthfulness(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = validate_layerwise(embeddings)
        return self._ensemble_probabilities(
            self.probes, values[:, self.summary.probe_layer, :]
        )

    def predict(self, embeddings: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_truthfulness(embeddings) >= threshold).astype(np.int64)

    def direct_truthfulness(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = validate_layerwise(embeddings)
        return self.summary.truth_score_sign * self.subspace.score(
            values[:, self.summary.subspace_layer, :]
        )

    def evaluate(self, embeddings: np.ndarray, truth_labels: np.ndarray) -> dict[str, float]:
        labels = np.asarray(truth_labels, dtype=np.int64).reshape(-1)
        probabilities = self.predict_truthfulness(embeddings)
        if len(labels) != len(probabilities) or len(np.unique(labels)) != 2:
            raise ValueError("evaluation needs matching labels containing both classes")
        return {
            "auroc": roc_auc(labels, probabilities),
            "average_precision": average_precision(labels, probabilities),
            "accuracy_at_0.5": binary_accuracy(labels, probabilities >= 0.5),
            "balanced_accuracy_at_0.5": balanced_accuracy(
                labels, probabilities >= 0.5
            ),
            "fpr_at_95_tpr": fpr_at_tpr(labels, probabilities),
            "brier_score": brier_score(labels, probabilities),
            "expected_calibration_error_10_bins": expected_calibration_error(
                labels, probabilities, bins=10
            ),
            "direct_projection_auroc": roc_auc(
                labels, self.direct_truthfulness(embeddings)
            ),
            "n_samples": len(labels),
        }

    def save(self, directory: str | Path) -> None:
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.subspace.save(directory / "subspace.npz")
        extension = "pt" if self.probe_config.backend == "torch_mlp" else "pkl"
        probe_files = []
        for index, probe in enumerate(self.probes):
            filename = f"probe_{index}.{extension}"
            probe.save(directory / filename)
            probe_files.append(filename)
        metadata = {
            "format_version": 2,
            "search": asdict(self.search),
            "probe_config": asdict(self.probe_config),
            "summary": asdict(self.summary),
            "probe_file": probe_files[0],
            "probe_files": probe_files,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> HaloScope:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        search_data = metadata["search"]
        for key in (
            "k_values",
            "threshold_quantiles",
            "layers",
            "probe_layers",
            "tail_fractions",
        ):
            if search_data.get(key) is not None:
                search_data[key] = tuple(search_data[key])
        result = cls(
            SearchConfig(**search_data),
            ProbeConfig(**metadata["probe_config"]),
        )
        result.summary = FitSummary(**metadata["summary"])
        result.subspace = LatentSubspace.load(directory / "subspace.npz")
        probe_files = metadata.get("probe_files", [metadata["probe_file"]])
        result.probes = [
            load_probe(directory / filename, result.probe_config.backend)
            for filename in probe_files
        ]
        result.probe = result.probes[0]
        return result

    def _truth_orientation(
        self, zeta: np.ndarray, truth_labels: np.ndarray
    ) -> tuple[int, float, float, float]:
        paper_selection, paper_auc, paper_std = self._validation_score(
            truth_labels, -zeta
        )
        if self.search.orientation == "paper":
            return -1, paper_selection, paper_auc, paper_std
        reverse_selection, reverse_auc, reverse_std = self._validation_score(
            truth_labels, zeta
        )
        if paper_selection >= reverse_selection:
            return -1, paper_selection, paper_auc, paper_std
        return 1, reverse_selection, reverse_auc, reverse_std

    def _validation_score(
        self, labels: np.ndarray, scores: np.ndarray
    ) -> tuple[float, float, float]:
        pooled_auc = roc_auc(labels, scores)
        if self.search.validation_folds == 1:
            return pooled_auc, pooled_auc, 0.0
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        fold_count = min(
            self.search.validation_folds, len(positive), len(negative)
        )
        if fold_count < 2:
            return pooled_auc, pooled_auc, 0.0
        rng = np.random.default_rng(self.probe_config.seed)
        positive = rng.permutation(positive)
        negative = rng.permutation(negative)
        fold_aucs = []
        for pos_fold, neg_fold in zip(
            np.array_split(positive, fold_count),
            np.array_split(negative, fold_count),
        ):
            indices = np.concatenate((pos_fold, neg_fold))
            fold_aucs.append(roc_auc(labels[indices], scores[indices]))
        fold_mean = float(np.mean(fold_aucs))
        fold_std = float(np.std(fold_aucs))
        selection_score = fold_mean - self.search.stability_penalty * fold_std
        return selection_score, pooled_auc, fold_std

    def _fit_probe_ensemble(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> list[TruthfulnessProbe]:
        probes = []
        for repeat in range(self.search.probe_repeats):
            config = replace(
                self.probe_config,
                seed=self.probe_config.seed + repeat,
                seed_each_fit=(
                    True
                    if self.search.probe_repeats > 1
                    else self.probe_config.seed_each_fit
                ),
            )
            probes.append(
                build_probe(config).fit(embeddings, labels, sample_weight=sample_weight)
            )
        return probes

    @staticmethod
    def _ensemble_probabilities(
        probes: list[TruthfulnessProbe], embeddings: np.ndarray
    ) -> np.ndarray:
        if not probes:
            raise RuntimeError("probe ensemble is empty")
        predictions = np.stack(
            [probe.predict_proba(embeddings) for probe in probes], axis=0
        )
        return predictions.mean(axis=0)

    def _pseudo_labels(
        self, scores: np.ndarray, quantile: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, float]:
        if self.search.pseudo_label_mode == "threshold":
            if not 0.0 < quantile < 1.0:
                raise ValueError("threshold quantiles must be strictly between 0 and 1")
            threshold = self._threshold(scores, quantile)
            selected = np.ones(len(scores), dtype=bool)
            labels = (scores > threshold).astype(np.int64)
            return selected, labels, None, threshold, threshold

        if not 0.0 < quantile <= 0.5:
            raise ValueError("tail fractions must be in (0, 0.5]")
        lower = self._threshold(scores, quantile)
        upper = self._threshold(scores, 1.0 - quantile)
        low = scores <= lower
        high = scores >= upper
        selected = low | high
        labels = high[selected].astype(np.int64)
        weights = None
        if self.search.confidence_weighted:
            order = np.argsort(scores, kind="mergesort")
            ranks = np.empty(len(scores), dtype=np.float64)
            ranks[order] = (np.arange(len(scores)) + 0.5) / len(scores)
            weights = (2.0 * np.abs(ranks - 0.5))[selected].astype(np.float32)
        return selected, labels, weights, lower, upper

    def _subspace_config(self, k: int) -> SubspaceConfig:
        return SubspaceConfig(
            k,
            weighted=self.search.weighted,
            center=self.search.center,
            score_centered=self.search.score_centered,
            score_mode=self.search.score_mode,
            deterministic_component_sign=self.search.deterministic_component_sign,
            factorization_backend=self.search.factorization_backend,
        )

    def _threshold(self, scores: np.ndarray, quantile: float) -> float:
        if self.search.quantile_method == "official":
            ordered = np.sort(scores)
            index = min(int(len(ordered) * quantile), len(ordered) - 1)
            return float(ordered[index])
        return float(np.quantile(scores, quantile))

    @staticmethod
    def _indices(
        requested: Iterable[int] | None, count: int, name: str
    ) -> tuple[int, ...]:
        values = tuple(range(count)) if requested is None else tuple(requested)
        if not values or any(value < 0 or value >= count for value in values):
            raise ValueError(f"{name} must contain indices in [0, {count - 1}]")
        return values

    def _require_fitted(self) -> None:
        if (
            self.subspace is None
            or self.probe is None
            or not self.probes
            or self.summary is None
        ):
            raise RuntimeError("HaloScope must be fitted before use")
