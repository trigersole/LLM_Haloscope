"""Dataset adapters for the four QA benchmarks used by the paper."""

from __future__ import annotations

from typing import Any


NO_CONTEXT_PROMPT = "Answer the question concisely. Q: {question} A:"
CONTEXT_PROMPT = (
    "Answer these questions concisely based on the context:\n"
    "Context: {context} Q: {question} A:"
)


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset preparation requires `datasets`. Install with `pip install -e .[llm]`."
        ) from exc
    return load_dataset


def _record(
    identifier: str,
    question: str,
    references: list[str],
    context: str | None = None,
) -> dict[str, Any]:
    references = list(dict.fromkeys(str(value).strip() for value in references if str(value).strip()))
    if not references:
        raise ValueError(f"sample {identifier} has no non-empty reference answer")
    prompt = (
        CONTEXT_PROMPT.format(context=context, question=question)
        if context
        else NO_CONTEXT_PROMPT.format(question=question)
    )
    return {
        "id": str(identifier),
        "question": question,
        "context": context,
        "prompt": prompt,
        "references": references,
    }


def load_benchmark(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Download and normalize a paper benchmark into one-record-per-generation format."""
    load_dataset = _require_datasets()
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in {"truthfulqa", "tqa"}:
        # `truthful_qa` was the legacy datasets-script identifier. The current
        # Hub dataset has a namespace, which recent `huggingface_hub` releases
        # require when resolving the dataset YAML/configuration.
        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        records = [
            _record(
                str(i),
                row["question"],
                [row["best_answer"], *row["correct_answers"]],
            )
            for i, row in enumerate(dataset)
        ]
    elif normalized == "triviaqa":
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        records, seen = [], set()
        for row in dataset:
            identifier = str(row["question_id"])
            if identifier in seen:
                continue
            seen.add(identifier)
            records.append(
                _record(identifier, row["question"], list(row["answer"]["aliases"]))
            )
    elif normalized in {"tydiqa", "tydiqagp"}:
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        records = []
        for row in dataset:
            if "english" not in str(row["id"]).lower():
                continue
            records.append(
                _record(
                    row["id"],
                    row["question"],
                    list(row["answers"]["text"]),
                    row["context"],
                )
            )
    elif normalized == "coqa":
        dataset = load_dataset("stanfordnlp/coqa", split="validation")
        records = []
        for story_index, row in enumerate(dataset):
            questions = row["questions"]
            answers = row["answers"]["input_text"]
            for turn, (question, answer) in enumerate(zip(questions, answers, strict=True)):
                records.append(
                    _record(
                        f"{row.get('id', story_index)}_{turn}",
                        question,
                        [answer],
                        row["story"],
                    )
                )
    else:
        raise ValueError("dataset must be one of: truthfulqa, triviaqa, coqa, tydiqa")
    return records if limit is None else records[:limit]
