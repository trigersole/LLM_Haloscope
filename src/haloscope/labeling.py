"""Reference-based truth labels used only for validation and evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np


def normalize_text(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Dependency-free ROUGE-L F1 for the paper's lightweight labeling alternative."""
    predicted = normalize_text(prediction)
    expected = normalize_text(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for column, expected_token in enumerate(expected, start=1):
            if token == expected_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


class SimilarityScorer(Protocol):
    def score(self, predictions: Sequence[str], references: Sequence[str]) -> np.ndarray: ...


class RougeLScorer:
    def score(self, predictions: Sequence[str], references: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [rouge_l_f1(prediction, reference) for prediction, reference in zip(
                predictions, references, strict=True
            )],
            dtype=np.float64,
        )


class BleurtScorer:
    """BLEURT via a Hugging Face sequence-classification checkpoint."""

    def __init__(
        self,
        model_name: str = "lucadiliello/BLEURT-20",
        *,
        batch_size: int = 16,
        device: str = "auto",
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("BLEURT scoring requires the [llm] dependencies") from exc
        self.torch = torch
        self.batch_size = batch_size
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        try:
            from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BLEURT requires bleurt-pytorch and Transformers v4. "
                "Run `uv sync --extra llm --extra bleurt` after pulling the latest project, "
                "or install `transformers>=4.41,<5` and `bleurt-pytorch` explicitly."
            ) from exc

        self.tokenizer = BleurtTokenizer.from_pretrained(model_name)
        self.model = BleurtForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, predictions: Sequence[str], references: Sequence[str]) -> np.ndarray:
        values = []
        for start in range(0, len(predictions), self.batch_size):
            end = start + self.batch_size
            encoded = self.tokenizer(
                list(references[start:end]),
                list(predictions[start:end]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.squeeze(-1)
            values.extend(logits.float().cpu().tolist())
        return np.asarray(values, dtype=np.float64)


def label_records(
    records: list[dict],
    scorer: SimilarityScorer,
    threshold: float = 0.5,
) -> list[dict]:
    """Attach max-reference similarity and truth_label (1 truthful, 0 hallucinated)."""
    output = []
    for record in records:
        references = record.get("references") or []
        if not references or "answer" not in record:
            raise ValueError("each record needs an answer and at least one reference")
        predictions = [record["answer"]] * len(references)
        scores = scorer.score(predictions, references)
        best = float(np.max(scores))
        output.append({**record, "similarity": best, "truth_label": int(best > threshold)})
    return output
