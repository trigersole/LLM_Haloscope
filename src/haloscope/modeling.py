"""Generic Hugging Face generation and activation extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_DIAGNOSTIC_TEMPLATE = """{context_block}Question: {question}
Proposed answer: {answer}

Consider whether the proposed answer is factually correct.
Assessment:"""

DEFAULT_CONTRASTIVE_POSITIVE_TEMPLATE = """{context_block}Question: {question}
Proposed answer: {answer}

Consider whether the proposed answer is factually correct.
Assessment:"""

DEFAULT_CONTRASTIVE_NEGATIVE_TEMPLATE = """{context_block}Question: {question}
Proposed answer: {answer}

Consider whether the proposed answer is factually incorrect.
Assessment:"""


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    representation: str = "block"
    dtype: str = "auto"
    device_map: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    batch_size: int = 1
    max_input_tokens: int = 2048
    max_new_tokens: int = 64
    num_beams: int = 5
    # The released TruthfulQA code removes a newly generated copy of its prompt
    # instruction before saving/scoring the answer. Empty by default so other
    # datasets retain their complete decoded output.
    answer_truncation_markers: tuple[str, ...] = ()
    strip_generated_answer: bool = True
    # response: the paper's prompt+generated-answer final-token representation.
    # diagnostic: the pre-verdict state of one factuality assessment prompt.
    # contrastive: positive-prompt state minus negative-prompt state.
    # prompt: the pre-generation state at the end of the original prompt.
    # endpoint_delta: response final state minus the prompt final state.
    # trajectory: prompt state plus answer-prefix/final changes from that state.
    activation_mode: str = "response"
    diagnostic_template: str = DEFAULT_DIAGNOSTIC_TEMPLATE
    contrastive_positive_template: str = DEFAULT_CONTRASTIVE_POSITIVE_TEMPLATE
    contrastive_negative_template: str = DEFAULT_CONTRASTIVE_NEGATIVE_TEMPLATE
    trajectory_checkpoints: tuple[int, ...] = (1, 4, 8)


def render_activation_prompt(template: str, record: dict) -> str:
    """Render a diagnostic prompt without asking the model to generate a verdict."""
    if "answer" not in record:
        raise ValueError("diagnostic activation extraction requires an answer")
    context = str(record.get("context") or "").strip()
    values = {
        "question": str(record.get("question") or "").strip(),
        "answer": str(record["answer"]).strip(),
        "context": context,
        "context_block": f"Context: {context}\n" if context else "",
        "prompt": str(record.get("prompt") or ""),
    }
    try:
        rendered = template.format_map(values)
    except KeyError as exc:
        allowed = ", ".join(sorted(values))
        raise ValueError(
            f"unknown activation-prompt placeholder {exc.args[0]!r}; allowed: {allowed}"
        ) from exc
    if not rendered.strip():
        raise ValueError("activation prompt rendered to an empty string")
    return rendered


class HFActivationModel:
    """Load an open-weight causal LM and return one last-token vector per transformer layer."""

    def __init__(self, config: ModelConfig):
        if config.representation not in {"block", "mlp", "attention"}:
            raise ValueError("representation must be block, mlp, or attention")
        supported_modes = {
            "response",
            "diagnostic",
            "contrastive",
            "prompt",
            "endpoint_delta",
            "trajectory",
        }
        if config.activation_mode not in supported_modes:
            raise ValueError(
                "activation_mode must be response, diagnostic, contrastive, prompt, "
                "endpoint_delta, or trajectory"
            )
        self.trajectory_checkpoints = tuple(int(value) for value in config.trajectory_checkpoints)
        if (
            not self.trajectory_checkpoints
            or any(value < 1 for value in self.trajectory_checkpoints)
            or tuple(sorted(set(self.trajectory_checkpoints))) != self.trajectory_checkpoints
        ):
            raise ValueError(
                "trajectory_checkpoints must be a non-empty, sorted list of unique "
                "positive token counts"
            )
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model execution requires PyTorch/Transformers. Install `pip install -e .[llm]`."
            ) from exc
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if config.device_map == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "model.device_map=cuda was requested, but CUDA is unavailable to PyTorch"
                )
            device_map: Any = {"": 0}
        else:
            device_map = config.device_map
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.attn_implementation is not None:
            kwargs["attn_implementation"] = config.attn_implementation
        if config.dtype != "auto":
            # The authors used Transformers 4.42.3, whose public argument is
            # torch_dtype. Newer v4 releases renamed it to dtype.
            major, minor = (int(value) for value in transformers.__version__.split(".")[:2])
            key = "torch_dtype" if (major, minor) < (4, 56) else "dtype"
            kwargs[key] = getattr(torch, config.dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
        self.model.eval()
        self.input_device = self._input_device()
        print(
            f"Loaded {config.model_name} on {self.input_device}; "
            f"CUDA allocated={torch.cuda.memory_allocated(0) / 2**30:.2f} GiB"
            if torch.cuda.is_available()
            else f"Loaded {config.model_name} on {self.input_device}; CUDA unavailable"
        )
        if config.device_map == "cuda" and self.input_device.type != "cuda":
            raise RuntimeError(
                f"OPT was required on CUDA but loaded on {self.input_device}"
            )

    def _input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        for parameter in self.model.parameters():
            if str(parameter.device) != "meta":
                return parameter.device
        return self.torch.device("cpu")

    def generate(self, prompts: list[str]) -> list[str]:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
            return_tensors="pt",
        ).to(self.input_device)
        with self.torch.inference_mode():
            sequences = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            sequences[:, prompt_width:], skip_special_tokens=True
        )

    def extract(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens + self.config.max_new_tokens,
            return_tensors="pt",
        ).to(self.input_device)
        if self.config.representation == "block":
            with self.torch.inference_mode():
                output = self.model(**encoded, output_hidden_states=True, use_cache=False)
            # hidden_states[0] is the token embedding, not a transformer block.
            states = list(output.hidden_states[1:])
            last_positions = self._last_non_padding(encoded["attention_mask"])
            vectors = []
            for state in states:
                positions = last_positions.to(state.device)
                batch_indices = self.torch.arange(len(texts), device=state.device)
                vectors.append(
                    state[batch_indices, positions, :].detach().float().cpu().numpy()
                )
        else:
            # Hooks retain only the selected token, avoiding all-layer sequence tensors in memory.
            vectors = self._extract_hooked(encoded)
        return np.stack(vectors, axis=1).astype(np.float32)

    def generate_and_extract(self, records: list[dict]) -> tuple[list[dict], np.ndarray]:
        completed: list[dict] = []
        activations = []
        size = self.config.batch_size
        for start in range(0, len(records), size):
            batch = records[start : start + size]
            answers = self.generate([record["prompt"] for record in batch])
            completed_batch = [
                {**record, "answer": self._postprocess_generated_answer(answer)}
                for record, answer in zip(batch, answers, strict=True)
            ]
            activations.append(self.extract_records(completed_batch))
            completed.extend(completed_batch)
        return completed, np.concatenate(activations, axis=0)

    def _postprocess_generated_answer(self, answer: str) -> str:
        """Apply configured dataset-specific cleanup to decoded model output."""
        for marker in self.config.answer_truncation_markers:
            if marker in answer:
                answer = answer.split(marker, maxsplit=1)[0]
        return answer.strip() if self.config.strip_generated_answer else answer

    def extract_records(self, records: list[dict]) -> np.ndarray:
        """Extract the configured endpoint or decoding-trajectory representation."""
        if not records:
            raise ValueError("at least one record is required for activation extraction")
        mode = self.config.activation_mode
        if mode == "response":
            texts = []
            for record in records:
                if "prompt" not in record or "answer" not in record:
                    raise ValueError("response activation extraction needs prompt and answer")
                texts.append(str(record["prompt"]) + str(record["answer"]))
            return self.extract(texts)
        if mode in {"prompt", "endpoint_delta", "trajectory"}:
            prompts, answers = self._prompt_answer_texts(records)
            prompt_states = self.extract(prompts)
            if mode == "prompt":
                return prompt_states

            response_texts = [
                prompt + answer for prompt, answer in zip(prompts, answers, strict=True)
            ]
            final_states = self.extract(response_texts)
            if mode == "endpoint_delta":
                return (final_states - prompt_states).astype(np.float32, copy=False)

            # Keep a fixed feature width for every sample. Each checkpoint is the
            # change from the pre-generation state after observing its first k
            # answer tokens. The final delta captures the completed response.
            pieces = [prompt_states]
            tokenized_answers = [
                self.tokenizer.encode(answer, add_special_tokens=False) for answer in answers
            ]
            for checkpoint in self.trajectory_checkpoints:
                prefix_texts = []
                for prompt, answer, token_ids in zip(
                    prompts, answers, tokenized_answers, strict=True
                ):
                    if checkpoint >= len(token_ids):
                        prefix = answer
                    else:
                        prefix = self.tokenizer.decode(
                            token_ids[:checkpoint],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    prefix_texts.append(prompt + prefix)
                checkpoint_states = self.extract(prefix_texts)
                pieces.append(checkpoint_states - prompt_states)
            pieces.append(final_states - prompt_states)
            return np.concatenate(pieces, axis=2).astype(np.float32, copy=False)
        if mode == "diagnostic":
            texts = [
                render_activation_prompt(self.config.diagnostic_template, record)
                for record in records
            ]
            return self.extract(texts)

        positive_texts = [
            render_activation_prompt(self.config.contrastive_positive_template, record)
            for record in records
        ]
        negative_texts = [
            render_activation_prompt(self.config.contrastive_negative_template, record)
            for record in records
        ]
        positive = self.extract(positive_texts)
        negative = self.extract(negative_texts)
        if positive.shape != negative.shape:
            raise RuntimeError(
                "contrastive prompt activations have different shapes: "
                f"{positive.shape} vs {negative.shape}"
            )
        return (positive - negative).astype(np.float32, copy=False)

    @staticmethod
    def _prompt_answer_texts(records: list[dict]) -> tuple[list[str], list[str]]:
        prompts = []
        answers = []
        for record in records:
            if "prompt" not in record or "answer" not in record:
                raise ValueError(
                    "prompt, endpoint_delta, and trajectory extraction need prompt and answer"
                )
            prompts.append(str(record["prompt"]))
            answers.append(str(record["answer"]))
        return prompts, answers

    def _last_non_padding(self, attention_mask):
        positions = self.torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        return (positions * attention_mask).max(dim=1).values

    def _layers(self):
        candidates = [
            ("model", "layers"),
            ("model", "decoder", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        ]
        for path in candidates:
            current = self.model
            try:
                for part in path:
                    current = getattr(current, part)
                return list(current)
            except AttributeError:
                continue
        raise RuntimeError("unsupported architecture: could not locate transformer layers")

    def _hook_module(self, layer):
        if self.config.representation == "attention":
            for name in ("self_attn", "attention", "attn"):
                if hasattr(layer, name):
                    return getattr(layer, name)
        else:
            if hasattr(layer, "mlp"):
                return layer.mlp
            if hasattr(layer, "fc2"):  # OPT feed-forward output projection
                return layer.fc2
        raise RuntimeError(
            f"unsupported layer {type(layer).__name__} for {self.config.representation} hooks"
        )

    def _extract_hooked(self, encoded):
        captured: list[Any] = [None] * len(self._layers())
        handles = []
        last_positions = self._last_non_padding(encoded["attention_mask"])
        batch_size, sequence_length = encoded["input_ids"].shape

        def make_hook(index):
            def hook(_module, _inputs, output):
                state = output[0] if isinstance(output, tuple) else output
                positions = last_positions.to(state.device)
                if state.ndim == 3:
                    batch_indices = self.torch.arange(batch_size, device=state.device)
                    selected = state[batch_indices, positions, :]
                elif state.ndim == 2 and state.shape[0] == batch_size * sequence_length:
                    # OPT flattens [batch, sequence, hidden] before its fc1/fc2 MLP
                    # projections, then reshapes it only after fc2 returns. A forward
                    # hook on fc2 therefore receives [batch * sequence, hidden].
                    flat_indices = (
                        self.torch.arange(batch_size, device=state.device) * sequence_length
                        + positions
                    )
                    selected = state[flat_indices, :]
                else:
                    raise RuntimeError(
                        "Unexpected hooked activation shape "
                        f"{tuple(state.shape)} for batch={batch_size}, sequence={sequence_length}"
                    )
                captured[index] = (
                    selected.detach().float().cpu().numpy()
                )

            return hook

        try:
            for index, layer in enumerate(self._layers()):
                handles.append(self._hook_module(layer).register_forward_hook(make_hook(index)))
            with self.torch.inference_mode():
                self.model(**encoded, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in captured):
            raise RuntimeError("one or more activation hooks did not run")
        return captured
