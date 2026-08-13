"""Generic Hugging Face generation and activation extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    representation: str = "block"
    dtype: str = "auto"
    device_map: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    batch_size: int = 1
    max_input_tokens: int = 2048
    max_new_tokens: int = 64
    num_beams: int = 5


class HFActivationModel:
    """Load an open-weight causal LM and return one last-token vector per transformer layer."""

    def __init__(self, config: ModelConfig):
        if config.representation not in {"block", "mlp", "attention"}:
            raise ValueError("representation must be block, mlp, or attention")
        try:
            import torch
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
        if config.dtype != "auto":
            kwargs["dtype"] = getattr(torch, config.dtype)
        else:
            kwargs["dtype"] = "auto"
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
            texts = [
                record["prompt"] + answer for record, answer in zip(batch, answers, strict=True)
            ]
            activations.append(self.extract(texts))
            completed.extend(
                {**record, "answer": answer.strip()}
                for record, answer in zip(batch, answers, strict=True)
            )
        return completed, np.concatenate(activations, axis=0)

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
