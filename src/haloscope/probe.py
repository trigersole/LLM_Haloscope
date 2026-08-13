"""Truthfulness probes used after HaloScope pseudo-label generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
import pickle

import numpy as np


@dataclass(frozen=True)
class ProbeConfig:
    backend: str = "torch_mlp"
    hidden_dim: int = 1024
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 0.05
    weight_decay: float = 3e-4
    seed: int = 41
    device: str = "auto"


class TruthfulnessProbe(Protocol):
    config: ProbeConfig

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TruthfulnessProbe": ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...


def build_probe(config: ProbeConfig) -> TruthfulnessProbe:
    if config.backend == "torch_mlp":
        return TorchMLPProbe(config)
    if config.backend == "logistic":
        return LogisticProbe(config)
    raise ValueError(f"unknown probe backend: {config.backend}")


def load_probe(path: str | Path, backend: str) -> TruthfulnessProbe:
    if backend == "torch_mlp":
        return TorchMLPProbe.load(path)
    if backend == "logistic":
        return LogisticProbe.load(path)
    raise ValueError(f"unknown probe backend: {backend}")


def _validate_training_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("probe data must have shapes [samples, features] and [samples]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("pseudo-labels must contain both classes 0 and 1")
    return x, y


class LogisticProbe:
    """Fast CPU probe for laptop checks; the full profile uses the paper's MLP."""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        x, y = _validate_training_data(x, y)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        standardized = (x - self.mean_) / self.scale_
        self.weights_ = np.zeros(x.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        # Full-batch gradient descent is deterministic and sufficient for the laptop/smoke probe.
        iterations = max(100, self.config.epochs * 20)
        learning_rate = min(0.2, max(0.01, self.config.learning_rate))
        targets = y.astype(np.float64)
        for step in range(iterations):
            logits = standardized @ self.weights_ + self.bias_
            probabilities = _sigmoid(logits)
            error = probabilities - targets
            decay = self.config.weight_decay * self.weights_
            gradient = standardized.T @ error / len(x) + decay
            bias_gradient = float(error.mean())
            rate = learning_rate * 0.5 * (1.0 + np.cos(np.pi * step / iterations))
            self.weights_ -= rate * gradient
            self.bias_ -= rate * bias_gradient
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("probe must be fitted before prediction")
        x = np.asarray(x, dtype=np.float64)
        return _sigmoid(((x - self.mean_) / self.scale_) @ self.weights_ + self.bias_)

    def save(self, path: str | Path) -> None:
        if self.weights_ is None:
            raise RuntimeError("cannot save an unfitted probe")
        with Path(path).open("wb") as handle:
            pickle.dump(
                {
                    "config": asdict(self.config),
                    "mean": self.mean_,
                    "scale": self.scale_,
                    "weights": self.weights_,
                    "bias": self.bias_,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticProbe":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        result = cls(ProbeConfig(**state["config"]))
        result.mean_ = state["mean"]
        result.scale_ = state["scale"]
        result.weights_ = state["weights"]
        result.bias_ = float(state["bias"])
        return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


class TorchMLPProbe:
    """The paper's 2-layer, ReLU, 1,024-hidden-unit classifier."""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model = None
        self.input_dim: int | None = None
        self.device_: str | None = None

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for backend=torch_mlp. Install with `pip install -e .[llm]`."
            ) from exc
        return torch

    def _resolve_device(self, torch) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _new_model(self, torch, input_dim: int):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_dim, 1),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPProbe":
        x, y = _validate_training_data(x, y)
        torch = self._torch()
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.input_dim = x.shape[1]
        self.device_ = self._resolve_device(torch)
        self.model = self._new_model(torch, self.input_dim).to(self.device_)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.config.epochs)
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.config.epochs):
            self.model.train()
            for features, labels in loader:
                features = features.to(self.device_)
                labels = labels.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(features).squeeze(-1)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("probe must be fitted before prediction")
        torch = self._torch()
        self.model.eval()
        with torch.inference_mode():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
            return torch.sigmoid(self.model(tensor).squeeze(-1)).cpu().numpy()

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted probe")
        torch = self._torch()
        torch.save(
            {
                "config": asdict(self.config),
                "input_dim": self.input_dim,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TorchMLPProbe":
        temporary = cls(ProbeConfig())
        torch = temporary._torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(ProbeConfig(**state["config"]))
        result.input_dim = int(state["input_dim"])
        result.device_ = result._resolve_device(torch)
        result.model = result._new_model(torch, result.input_dim)
        result.model.load_state_dict(state["state_dict"])
        result.model.to(result.device_)
        result.model.eval()
        return result
