"""Command-line interface for resumable laptop and remote HaloScope experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import load_config, probe_config, search_config, work_paths
from .data import load_benchmark
from .labeling import BleurtScorer, RougeLScorer, label_records
from .modeling import HFActivationModel, ModelConfig
from .pipeline import HaloScope, SearchConfig
from .probe import ProbeConfig
from .records import read_jsonl, write_jsonl
from .splitting import DataSplit, make_split


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _model_config(config: dict) -> ModelConfig:
    values = dict(config.get("model", {}))
    if "model_name" not in values:
        raise ValueError("config.model.model_name is required")
    return ModelConfig(**values)


def _activation_metadata(config: dict) -> dict:
    model = _model_config(config)
    metadata = {
        "activation_mode": model.activation_mode,
        "representation": model.representation,
        "diagnostic_template": model.diagnostic_template,
        "contrastive_positive_template": model.contrastive_positive_template,
        "contrastive_negative_template": model.contrastive_negative_template,
        "contrastive_operation": "positive_minus_negative",
    }
    if model.answer_truncation_markers or not model.strip_generated_answer:
        metadata["answer_truncation_markers"] = list(model.answer_truncation_markers)
        metadata["strip_generated_answer"] = model.strip_generated_answer
    if model.attn_implementation is not None:
        metadata["attn_implementation"] = model.attn_implementation
    # Do not change metadata for the original modes: existing checkpoints must
    # remain resumable and exactly reproducible after trajectory support is added.
    if model.activation_mode == "trajectory":
        metadata["trajectory_checkpoints"] = list(model.trajectory_checkpoints)
        metadata["trajectory_features"] = [
            "prompt_state",
            *[f"delta_after_{value}_answer_tokens" for value in model.trajectory_checkpoints],
            "final_state_minus_prompt_state",
        ]
    elif model.activation_mode == "endpoint_delta":
        metadata["endpoint_delta_operation"] = "final_state_minus_prompt_state"
    return metadata


def _write_activation_metadata(config: dict, path: Path, source: str) -> None:
    metadata = {**_activation_metadata(config), "source": source}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def command_prepare(config: dict) -> None:
    paths = work_paths(config)
    dataset = config.get("dataset", {})
    examples = load_benchmark(dataset.get("name", "truthfulqa"), dataset.get("limit"))
    write_jsonl(paths["examples"], examples)
    print(f"Prepared {len(examples)} examples -> {paths['examples']}")


def command_generate(config: dict) -> None:
    paths = work_paths(config)
    examples = read_jsonl(paths["examples"])
    completed = read_jsonl(paths["generations"]) if paths["generations"].exists() else []
    if completed:
        if not paths["embeddings"].exists():
            raise RuntimeError("generation checkpoint exists but embeddings.npy is missing")
        activations = list(np.load(paths["embeddings"]))
        expected = [record["id"] for record in examples[: len(completed)]]
        if [record["id"] for record in completed] != expected:
            raise RuntimeError("generation checkpoint is not a prefix of examples.jsonl")
        if paths["activation_metadata"].exists():
            recorded = json.loads(paths["activation_metadata"].read_text(encoding="utf-8"))
            expected_metadata = _activation_metadata(config)
            mismatched = [
                key for key, value in expected_metadata.items() if recorded.get(key) != value
            ]
            if mismatched:
                raise RuntimeError(
                    "saved generation/embeddings use different settings "
                    f"({', '.join(mismatched)}); use a fresh work_dir to regenerate them"
                )
    else:
        activations = []
    if len(completed) == len(examples):
        print(f"Generation already complete ({len(completed)} samples)")
        return

    seed = int(config.get("seed", 41))
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    model = HFActivationModel(_model_config(config))
    checkpoint_every = int(config.get("checkpoint_every", 10))
    for start in range(len(completed), len(examples), checkpoint_every):
        batch = examples[start : start + checkpoint_every]
        new_records, new_activations = model.generate_and_extract(batch)
        completed.extend(new_records)
        activations.extend(new_activations)
        write_jsonl(paths["generations"], completed)
        _atomic_numpy(paths["embeddings"], np.asarray(activations, dtype=np.float32))
        _write_activation_metadata(config, paths["activation_metadata"], "generation")
        print(f"Checkpoint: {len(completed)}/{len(examples)} generations")


def command_extract(config: dict, source_config: dict | None = None) -> None:
    """Rebuild activations from saved answers, optionally into a new experiment."""
    paths = work_paths(config)
    source_paths = work_paths(source_config) if source_config is not None else paths
    records = read_jsonl(source_paths["generations"])
    if not records:
        raise RuntimeError("no saved generations are available for extraction")
    if source_paths["generations"] != paths["generations"]:
        write_jsonl(paths["generations"], records)
        if source_paths["examples"].exists():
            write_jsonl(paths["examples"], read_jsonl(source_paths["examples"]))

    source_name = str(source_paths["generations"].resolve())
    expected_metadata = {**_activation_metadata(config), "source": source_name}
    activations = []
    if paths["embeddings"].exists() and paths["activation_metadata"].exists():
        recorded = json.loads(paths["activation_metadata"].read_text(encoding="utf-8"))
        if all(recorded.get(key) == value for key, value in expected_metadata.items()):
            saved = np.load(paths["embeddings"])
            if len(saved) > len(records):
                raise RuntimeError("saved embeddings contain more records than generations")
            activations = list(saved)

    model_config = _model_config(config)
    batch_size = max(1, model_config.batch_size)
    checkpoint_every = max(batch_size, int(config.get("checkpoint_every", 10)))
    last_checkpoint = len(activations)
    if last_checkpoint == len(records):
        print(f"Extraction already complete ({len(records)} samples)")
        return
    model = HFActivationModel(model_config)
    for start in range(last_checkpoint, len(records), batch_size):
        batch = records[start : start + batch_size]
        activations.extend(model.extract_records(batch))
        completed = start + len(batch)
        if completed - last_checkpoint >= checkpoint_every or completed == len(records):
            _atomic_numpy(paths["embeddings"], np.asarray(activations, dtype=np.float32))
            _write_activation_metadata(
                config,
                paths["activation_metadata"],
                source_name,
            )
            last_checkpoint = completed
            print(f"Extraction checkpoint: {completed}/{len(records)}")


def command_label(config: dict) -> None:
    paths = work_paths(config)
    records = read_jsonl(paths["generations"])
    options = config.get("labeling", {})
    metric = options.get("metric", "rouge_l").lower()
    if metric == "rouge_l":
        scorer = RougeLScorer()
    elif metric == "bleurt":
        scorer = BleurtScorer(
            options.get("model_name", "lucadiliello/BLEURT-20"),
            batch_size=int(options.get("batch_size", 16)),
            device=options.get("device", "auto"),
            input_order=options.get("input_order", "reference_candidate"),
        )
    else:
        raise ValueError("labeling.metric must be rouge_l or bleurt")
    labeled = label_records(records, scorer, float(options.get("threshold", 0.5)))
    write_jsonl(paths["labeled"], labeled)
    truthful = sum(record["truth_label"] for record in labeled)
    print(f"Labeled {len(labeled)} samples: {truthful} truthful, {len(labeled)-truthful} hallucinated")


def _load_or_make_split(config: dict, n_samples: int, path: Path) -> DataSplit:
    if path.exists():
        with np.load(path) as values:
            split = DataSplit(values["wild"], values["validation"], values["test"])
        split.validate(n_samples)
        return split
    options = config.get("split", {})
    split = make_split(
        n_samples,
        wild_ratio=float(options.get("wild_ratio", 0.75)),
        validation_size=int(options.get("validation_size", 100)),
        seed=int(config.get("seed", 41)),
    )
    np.savez_compressed(path, wild=split.wild, validation=split.validation, test=split.test)
    return split


def command_train(config: dict) -> None:
    paths = work_paths(config)
    embeddings = np.load(paths["embeddings"])
    records = read_jsonl(paths["labeled"])
    if len(embeddings) != len(records):
        raise RuntimeError("embeddings and labeled records have different sample counts")
    labels = np.asarray([record["truth_label"] for record in records], dtype=np.int64)
    split = _load_or_make_split(config, len(records), paths["split"])
    search = search_config(config)
    if search.replay_official_numpy_rng:
        # The released process seeds NumPy once, generates this permutation, and
        # then lets sklearn PCA(random_state=None) consume the advanced stream.
        np.random.seed(int(config.get("seed", 41)))
        np.random.permutation(len(records))
    detector = HaloScope(search, probe_config(config)).fit(
        embeddings[split.wild],
        embeddings[split.validation],
        labels[split.validation],
    )
    detector.save(paths["detector"])
    print(json.dumps(detector.summary.__dict__, indent=2))
    print(f"Saved detector -> {paths['detector']}")


def command_evaluate(config: dict) -> dict[str, float]:
    paths = work_paths(config)
    embeddings = np.load(paths["embeddings"])
    records = read_jsonl(paths["labeled"])
    labels = np.asarray([record["truth_label"] for record in records], dtype=np.int64)
    split = _load_or_make_split(config, len(records), paths["split"])
    detector = HaloScope.load(paths["detector"])
    metrics = detector.evaluate(embeddings[split.test], labels[split.test])
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def command_score(
    config: dict,
    prompt: str,
    answer: str,
    question: str | None = None,
    context: str | None = None,
) -> None:
    paths = work_paths(config)
    model = HFActivationModel(_model_config(config))
    embedding = model.extract_records(
        [
            {
                "prompt": prompt,
                "question": question if question is not None else prompt,
                "context": context,
                "answer": answer,
            }
        ]
    )
    probability = float(HaloScope.load(paths["detector"]).predict_truthfulness(embedding)[0])
    print(json.dumps({"truthfulness": probability, "hallucination": 1.0 - probability}, indent=2))


def command_smoke(output: str | None = None) -> dict[str, float]:
    """Run the complete algorithm on synthetic activations with no model download."""
    rng = np.random.default_rng(41)
    n, layers, hidden = 240, 6, 32
    labels = (rng.random(n) > 0.30).astype(np.int64)  # 1 truthful, 0 hallucinated
    embeddings = rng.normal(0, 0.55, size=(n, layers, hidden)).astype(np.float32)
    anomaly = rng.normal(5.0, 0.5, size=(int((labels == 0).sum()), 1))
    embeddings[labels == 0, 2, :3] += anomaly
    embeddings[labels == 0, 3, :3] += anomaly * 0.7
    split = make_split(n, validation_size=40, seed=41)
    detector = HaloScope(
        SearchConfig(
            k_values=(1, 2, 3, 4),
            threshold_quantiles=(0.2, 0.3, 0.4, 0.5, 0.6),
            orientation="paper",
        ),
        ProbeConfig(backend="logistic", epochs=20, seed=41),
    ).fit(
        embeddings[split.wild],
        embeddings[split.validation],
        labels[split.validation],
    )
    metrics = detector.evaluate(embeddings[split.test], labels[split.test])
    if output:
        detector.save(output)
    print(json.dumps({"selection": detector.summary.__dict__, "metrics": metrics}, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="haloscope", description="HaloScope NeurIPS 2024 reimplementation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "generate", "label", "train", "evaluate", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="YAML experiment configuration")
    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", required=True, help="target experiment configuration")
    extract.add_argument(
        "--source-config",
        help="optional experiment config whose saved generations should be reused",
    )
    score = subparsers.add_parser("score")
    score.add_argument("--config", required=True)
    score.add_argument("--prompt", required=True)
    score.add_argument("--answer", required=True)
    score.add_argument(
        "--question",
        help="raw question used by diagnostic/contrastive activation prompts",
    )
    score.add_argument(
        "--context",
        help="optional source context used by diagnostic/contrastive activation prompts",
    )
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", help="optional detector output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        command_smoke(args.output)
        return
    config = load_config(args.config)
    commands = {
        "prepare": command_prepare,
        "generate": command_generate,
        "label": command_label,
        "train": command_train,
        "evaluate": command_evaluate,
    }
    if args.command == "score":
        command_score(config, args.prompt, args.answer, args.question, args.context)
    elif args.command == "extract":
        source_config = load_config(args.source_config) if args.source_config else None
        command_extract(config, source_config)
    elif args.command == "all":
        for name in ("prepare", "generate", "label", "train", "evaluate"):
            commands[name](config)
    else:
        commands[args.command](config)


if __name__ == "__main__":
    main()
