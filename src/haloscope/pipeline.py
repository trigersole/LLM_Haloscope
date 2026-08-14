"""End-to-end HaloScope selection, pseudo-labeling, training, and inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .core import LatentSubspace, SubspaceConfig, validate_layerwise
from .metrics import average_precision, binary_accuracy, roc_auc
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
    selection_source: str = "wild"
    quantile_method: str = "linear"
    retrain_selected_probe: bool = False
    # "paper": high zeta is hallucinated; "auto": choose direction using validation labels,
    # reproducing the official code's sign search.
    orientation: str = "paper"


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
        self.summary: FitSummary | None = None

    def fit(
        self,
        wild_embeddings: np.ndarray,
        validation_embeddings: np.ndarray,
        validation_truth_labels: np.ndarray,
    ) -> "HaloScope":
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

        if self.probe_config.backend == "torch_mlp" and not self.probe_config.seed_each_fit:
            import torch

            torch.manual_seed(self.probe_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.probe_config.seed)

        layers = self._indices(self.search.layers, wild.shape[1], "layers")
        selection = validation if self.search.selection_source == "validation" else wild
        best_direct: tuple[float, int, int, int] | None = None
        # Released code iterates k first, then layer; preserve its tie-breaking.
        for k in self.search.k_values:
            for layer in layers:
                if k > min(selection.shape[0], selection.shape[2]):
                    continue
                candidate = LatentSubspace(
                    self._subspace_config(k)
                ).fit(selection[:, layer, :])
                zeta = candidate.score(validation[:, layer, :])
                sign, auc = self._truth_orientation(zeta, labels)
                if best_direct is None or auc > best_direct[0]:
                    best_direct = (auc, layer, k, sign)
        if best_direct is None:
            raise ValueError("no valid layer/k candidate; reduce k_values or add wild samples")

        direct_auc, subspace_layer, k, sign = best_direct
        self.subspace = LatentSubspace(
            self._subspace_config(k)
        ).fit(wild[:, subspace_layer, :])
        wild_truth_score = sign * self.subspace.score(wild[:, subspace_layer, :])
        probe_layers = self._indices(self.search.probe_layers, wild.shape[1], "probe_layers")

        best_probe = None
        for quantile in self.search.threshold_quantiles:
            if not 0.0 < quantile < 1.0:
                raise ValueError("threshold quantiles must be strictly between 0 and 1")
            threshold = self._threshold(wild_truth_score, quantile)
            pseudo_truth = (wild_truth_score > threshold).astype(np.int64)
            if len(np.unique(pseudo_truth)) != 2:
                continue
            for layer in probe_layers:
                candidate_probe = build_probe(self.probe_config).fit(
                    wild[:, layer, :], pseudo_truth
                )
                probabilities = candidate_probe.predict_proba(validation[:, layer, :])
                auc = roc_auc(labels, probabilities)
                if best_probe is None or auc > best_probe[0]:
                    best_probe = (
                        auc,
                        layer,
                        quantile,
                        threshold,
                        pseudo_truth.copy(),
                        candidate_probe,
                    )
        if best_probe is None:
            raise RuntimeError("probe search produced no valid model")

        auc, probe_layer, quantile, threshold, pseudo_truth, selected_probe = best_probe
        self.probe = (
            build_probe(self.probe_config).fit(wild[:, probe_layer, :], pseudo_truth)
            if self.search.retrain_selected_probe
            else selected_probe
        )
        self.summary = FitSummary(
            subspace_layer=subspace_layer,
            probe_layer=probe_layer,
            n_components=k,
            threshold_quantile=float(quantile),
            threshold_value=float(threshold),
            truth_score_sign=sign,
            validation_auroc=float(auc),
            validation_direct_auroc=float(direct_auc),
            pseudo_truthful=int(pseudo_truth.sum()),
            pseudo_hallucinated=int((1 - pseudo_truth).sum()),
        )
        return self

    def predict_truthfulness(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = validate_layerwise(embeddings)
        return self.probe.predict_proba(values[:, self.summary.probe_layer, :])

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
            "direct_projection_auroc": roc_auc(
                labels, self.direct_truthfulness(embeddings)
            ),
            "n_samples": int(len(labels)),
        }

    def save(self, directory: str | Path) -> None:
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.subspace.save(directory / "subspace.npz")
        extension = "pt" if self.probe_config.backend == "torch_mlp" else "pkl"
        self.probe.save(directory / f"probe.{extension}")
        metadata = {
            "format_version": 1,
            "search": asdict(self.search),
            "probe_config": asdict(self.probe_config),
            "summary": asdict(self.summary),
            "probe_file": f"probe.{extension}",
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "HaloScope":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        search_data = metadata["search"]
        for key in ("k_values", "threshold_quantiles", "layers", "probe_layers"):
            if search_data.get(key) is not None:
                search_data[key] = tuple(search_data[key])
        result = cls(
            SearchConfig(**search_data),
            ProbeConfig(**metadata["probe_config"]),
        )
        result.summary = FitSummary(**metadata["summary"])
        result.subspace = LatentSubspace.load(directory / "subspace.npz")
        result.probe = load_probe(
            directory / metadata["probe_file"], result.probe_config.backend
        )
        return result

    def _truth_orientation(self, zeta: np.ndarray, truth_labels: np.ndarray) -> tuple[int, float]:
        paper_auc = roc_auc(truth_labels, -zeta)
        if self.search.orientation == "paper":
            return -1, paper_auc
        reverse_auc = roc_auc(truth_labels, zeta)
        return (-1, paper_auc) if paper_auc >= reverse_auc else (1, reverse_auc)

    def _subspace_config(self, k: int) -> SubspaceConfig:
        return SubspaceConfig(
            k,
            weighted=self.search.weighted,
            center=self.search.center,
            score_centered=self.search.score_centered,
            score_mode=self.search.score_mode,
            deterministic_component_sign=self.search.deterministic_component_sign,
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
        if self.subspace is None or self.probe is None or self.summary is None:
            raise RuntimeError("HaloScope must be fitted before use")
