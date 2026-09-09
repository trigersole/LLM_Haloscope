"""Compare a resumable HaloScope run with the authors' released artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def official_answer_path(root: Path, model: str, dataset: str, index: int) -> Path:
    return (
        root
        / "save_for_eval"
        / f"{dataset}_hal_det"
        / "answers"
        / f"most_likely_hal_det_{model}_{dataset}_answers_index_{index}.npy"
    )


def compare_answers(ours: Path, official: Path, model: str, dataset: str) -> dict:
    generations = read_jsonl(ours / "generations.jsonl")
    mismatches = []
    missing = []
    equal = 0
    for index, row in enumerate(generations):
        path = official_answer_path(official, model, dataset, index)
        if not path.exists():
            missing.append(index)
            continue
        values = np.load(path, allow_pickle=True).reshape(-1)
        official_answer = "" if len(values) == 0 else str(values[0])
        if row["answer"] == official_answer:
            equal += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {
                    "index": index,
                    "ours": row["answer"],
                    "official": official_answer,
                }
            )
    return {
        "ours_count": len(generations),
        "official_found": len(generations) - len(missing),
        "exact_matches": equal,
        "exact_match_rate": equal / len(generations) if generations else None,
        "first_missing_indices": missing[:20],
        "first_mismatches": mismatches,
    }


def compare_labels(ours: Path, official: Path, dataset: str, threshold: float) -> dict:
    ours_path = ours / "labeled.jsonl"
    official_path = official / f"ml_{dataset}_bleurt_score.npy"
    if not ours_path.exists() or not official_path.exists():
        return {
            "available": False,
            "ours_path": str(ours_path),
            "official_path": str(official_path),
        }
    rows = read_jsonl(ours_path)
    ours_scores = np.asarray([row["similarity"] for row in rows], dtype=np.float64)
    official_scores = np.asarray(np.load(official_path), dtype=np.float64).reshape(-1)
    count = min(len(ours_scores), len(official_scores))
    difference = np.abs(ours_scores[:count] - official_scores[:count])
    ours_labels = ours_scores[:count] > threshold
    official_labels = official_scores[:count] > threshold
    disagreements = np.flatnonzero(ours_labels != official_labels)
    return {
        "available": True,
        "ours_count": len(ours_scores),
        "official_count": len(official_scores),
        "compared": count,
        "max_abs_score_difference": float(difference.max()) if count else None,
        "mean_abs_score_difference": float(difference.mean()) if count else None,
        "label_agreement_rate": float(np.mean(ours_labels == official_labels)) if count else None,
        "ours_truthful": int(ours_labels.sum()),
        "official_truthful": int(official_labels.sum()),
        "first_label_disagreements": disagreements[:20].tolist(),
    }


def compare_split(ours: Path, sample_count: int, seed: int, wild_ratio: float) -> dict:
    path = ours / "split.npz"
    if not path.exists():
        return {"available": False, "path": str(path)}
    permutation = np.random.RandomState(seed).permutation(sample_count)
    wild_and_validation = int(wild_ratio * sample_count)
    wild_count = wild_and_validation - 100
    expected = {
        "wild": np.sort(permutation[:wild_count]),
        "validation": np.sort(permutation[wild_count:wild_and_validation]),
        "test": np.sort(permutation[wild_and_validation:]),
    }
    with np.load(path) as saved:
        result = {
            name: {
                "count": len(saved[name]),
                "exact_match": bool(np.array_equal(saved[name], indices)),
            }
            for name, indices in expected.items()
        }
    return {"available": True, **result}


def compare_embeddings(ours: Path, official: Path, model: str, dataset: str) -> dict:
    ours_path = ours / "embeddings.npy"
    official_path = (
        official
        / "save_for_eval"
        / f"{dataset}_hal_det"
        / f"most_likely_{model}_gene_embeddings_layer_wise.npy"
    )
    if not ours_path.exists() or not official_path.exists():
        return {
            "available": False,
            "ours_path": str(ours_path),
            "official_path": str(official_path),
        }
    ours_values = np.load(ours_path, mmap_mode="r")
    official_values = np.load(official_path, mmap_mode="r")
    if official_values.ndim == 3 and official_values.shape[1] == ours_values.shape[1] + 1:
        official_values = official_values[:, 1:, :]
    result: dict[str, Any] = {
        "available": True,
        "ours_shape": list(ours_values.shape),
        "official_shape_after_block_alignment": list(official_values.shape),
    }
    if ours_values.shape != official_values.shape:
        result["shape_match"] = False
        return result
    result["shape_match"] = True
    layers = []
    total_absolute = 0.0
    total_values = 0
    overall_max = 0.0
    for layer in range(ours_values.shape[1]):
        left = np.asarray(ours_values[:, layer, :], dtype=np.float64)
        right = np.asarray(official_values[:, layer, :], dtype=np.float64)
        difference = np.abs(left - right)
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        denominator = np.maximum(left_norm * right_norm, 1e-30)
        cosine = np.sum(left * right, axis=1) / denominator
        layer_max = float(difference.max())
        overall_max = max(overall_max, layer_max)
        total_absolute += float(difference.sum())
        total_values += difference.size
        layers.append(
            {
                "layer": layer,
                "max_abs_difference": layer_max,
                "mean_abs_difference": float(difference.mean()),
                "mean_cosine_similarity": float(cosine.mean()),
            }
        )
    result.update(
        {
            "max_abs_difference": overall_max,
            "mean_abs_difference": total_absolute / total_values,
            "minimum_layer_mean_cosine": min(
                layer["mean_cosine_similarity"] for layer in layers
            ),
            "layers": layers,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, type=Path, help="Our experiment work_dir")
    parser.add_argument("--official", required=True, type=Path, help="Official repository root")
    parser.add_argument("--model", default="llama2_chat_7B")
    parser.add_argument("--dataset", default="tqa")
    parser.add_argument("--seed", default=41, type=int)
    parser.add_argument("--wild-ratio", default=0.75, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generations = read_jsonl(args.ours / "generations.jsonl")
    report = {
        "answers": compare_answers(args.ours, args.official, args.model, args.dataset),
        "labels": compare_labels(args.ours, args.official, args.dataset, args.threshold),
        "split": compare_split(args.ours, len(generations), args.seed, args.wild_ratio),
        "embeddings": compare_embeddings(args.ours, args.official, args.model, args.dataset),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
