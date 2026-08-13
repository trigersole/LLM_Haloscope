"""YAML experiment configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .pipeline import SearchConfig
from .probe import ProbeConfig


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path.resolve())
    return config


def search_config(config: dict[str, Any]) -> SearchConfig:
    values = dict(config.get("search", {}))
    for key in ("k_values", "threshold_quantiles", "layers", "probe_layers"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return SearchConfig(**values)


def probe_config(config: dict[str, Any]) -> ProbeConfig:
    return ProbeConfig(**config.get("probe", {}))


def work_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config.get("work_dir", "outputs/experiment"))
    return {
        "root": root,
        "examples": root / "examples.jsonl",
        "generations": root / "generations.jsonl",
        "embeddings": root / "embeddings.npy",
        "labeled": root / "labeled.jsonl",
        "detector": root / "detector",
        "split": root / "split.npz",
        "metrics": root / "metrics.json",
    }

