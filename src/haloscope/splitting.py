"""Deterministic paper-compatible train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataSplit:
    wild: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, n_samples: int) -> None:
        joined = np.concatenate([self.wild, self.validation, self.test])
        if len(joined) != n_samples or len(np.unique(joined)) != n_samples:
            raise ValueError("split indices must partition every sample exactly once")


def make_split(
    n_samples: int,
    *,
    wild_ratio: float = 0.75,
    validation_size: int = 100,
    seed: int = 41,
) -> DataSplit:
    """Match the official split: permute, take 75%, reserve its final 100 for validation."""
    if n_samples < 6:
        raise ValueError("at least 6 samples are required")
    if not 0.5 <= wild_ratio < 1:
        raise ValueError("wild_ratio must be in [0.5, 1)")
    permutation = np.random.default_rng(seed).permutation(n_samples)
    wild_and_validation_count = int(wild_ratio * n_samples)
    validation_size = min(validation_size, max(1, wild_and_validation_count // 3))
    wild_count = wild_and_validation_count - validation_size
    if wild_count < 2:
        raise ValueError("split leaves fewer than two unlabeled wild samples")
    result = DataSplit(
        wild=permutation[:wild_count],
        validation=permutation[wild_count:wild_and_validation_count],
        test=permutation[wild_and_validation_count:],
    )
    result.validate(n_samples)
    return result

