# HaloScope — clean NeurIPS 2024 reimplementation

This project implements **“HaloScope: Harnessing Unlabeled LLM Generations for Hallucination
Detection”** by Du, Xiao, and Li from scratch. It includes the complete paper pipeline, a small
laptop profile, and resumable remote-GPU scripts for SSH/PuTTY.

The isolated `codex/haloscope-plus` research branch adds confidence-aware
pseudo-labeling, compact multi-seed probes, fold-stable selection, and endpoint-delta
experiments. See [the HaloScope++ guide](docs/HALOSCOPE_PLUS.md). Existing configurations
retain their previous behavior.

- [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2024/file/ba92705991cfbbcedc26e27e833ebbae-Paper-Conference.pdf)
- [Authors' reference repository](https://github.com/deeplearning-wisc/haloscope)
- [Detailed PuTTY guide](docs/PUTTY_REMOTE.md)
- [Detailed Slurm guide](docs/SLURM.md)

## What is implemented

1. TruthfulQA, TriviaQA, CoQA, and TyDiQA-GP dataset adapters.
2. Greedy/beam answer generation from any compatible Hugging Face causal LM.
3. Last-token activation extraction from every block, attention output, or MLP output.
4. Centered SVD and the singular-value-weighted Equation 7 membership score.
5. Validation search over activation layer, number of components `k`, score orientation, pseudo-label
   threshold, and classifier layer.
6. The paper's two-layer ReLU MLP (hidden size 1,024), SGD, cosine decay, and binary sigmoid loss.
7. BLEURT or dependency-free ROUGE-L reference labels for validation/evaluation only.
8. AUROC, average precision, accuracy, and direct-projection baseline evaluation.
9. Atomic checkpoints, saved detectors, and single-answer inference.

HaloScope does **not** verify facts against the web. It learns a detector for one model's internal
activation distribution. A detector trained on LLaMA-2 activations cannot be applied to OPT, Qwen,
or another hidden size without retraining.

## Algorithm in one minute

For each unlabeled prompt/answer pair, the model supplies a last-token activation
`f_i ∈ R^d`. After centering all activations, SVD gives the top directions `v_j` and singular values
`σ_j`. The membership score is:

```text
ζ_i = (1/k) Σ_j σ_j <f_i - μ, v_j>²
```

The paper treats large `ζ` as likely hallucinated. Thresholding it creates noisy pseudo-labels, which
train a truthfulness MLP. Only the small validation and test partitions use reference-derived labels;
the wild training partition remains unlabeled.

## Fast check without downloading an LLM

The current Python environment only needs NumPy and scikit-learn:

```bash
python -m pip install -e '.[dev]'
haloscope smoke
pytest
```

The smoke command creates synthetic layer activations, runs SVD selection, pseudo-labeling, probe
training, evaluation, and artifact serialization.

## Laptop end-to-end version

The laptop config uses SmolLM2-135M, 120 TruthfulQA examples, ROUGE-L, a reduced hyperparameter
grid, and a logistic probe. It checks the complete data/model/activation pipeline but is not expected
to reproduce the paper's AUROC.

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_laptop.ps1
```

Or run each stage:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[llm]"
haloscope prepare  --config configs/laptop_smollm2.yaml
haloscope generate --config configs/laptop_smollm2.yaml
haloscope label    --config configs/laptop_smollm2.yaml
haloscope train    --config configs/laptop_smollm2.yaml
haloscope evaluate --config configs/laptop_smollm2.yaml
```

CPU generation may take a while; the generated sample count is checkpointed every ten records.
If the reference labels contain only one class in the small subset, increase `dataset.limit` or choose
a different `seed`.

## Full paper run on a remote GPU

Use an NVIDIA GPU server and follow [docs/PUTTY_REMOTE.md](docs/PUTTY_REMOTE.md). The short form is:

```bash
bash scripts/setup_remote.sh
tmux new -s haloscope
bash scripts/run_remote.sh configs/paper_exact_llama2_7b_truthfulqa.yaml
```

Available full profiles:

| Config | Base model | Paper representation | Access |
|---|---|---|---|
| `paper_exact_llama2_7b_truthfulqa.yaml` | LLaMA-2-7B-Chat | block output | gated |
| `paper_exact_opt_6.7b_truthfulqa.yaml` | OPT-6.7B | MLP output | open |
| `official_llama2_7b_truthfulqa.yaml` | LLaMA-2-7B-Chat | authors' released-code settings | gated |
| `official_opt_6.7b_truthfulqa.yaml` | OPT-6.7B | authors' released-code settings | open |

### Slurm cluster

After creating the remote environment, submit the included batch job:

```bash
mkdir -p logs
sbatch scripts/slurm_haloscope.sbatch \
  all configs/paper_exact_llama2_7b_truthfulqa.yaml
```

See [docs/SLURM.md](docs/SLURM.md) for partitions, accounts, monitoring, resuming, and common
cluster errors.

The profiles use all 817 TruthfulQA questions, seed 41, a 75% wild/validation partition, 100
validation questions, five-beam generation, BLEURT threshold 0.5, `k=1..10`, the full threshold
grid, and the paper's MLP settings.

## Score a new answer

Use the same model, representation, and prompt style as training:

```bash
haloscope score \
  --config configs/paper_exact_llama2_7b_truthfulqa.yaml \
  --prompt "Answer the question concisely. Q: Who wrote Hamlet? A:" \
  --answer "William Shakespeare."
```

The output contains truthfulness and hallucination scores that sum to one. They are detector
probabilities, not a guarantee of factual correctness.

## Prompt-contrast activation experiment

The prompt-contrast profiles keep answer generation and BLEURT evaluation identical to the
released-code profiles. For every saved answer, they run two additional forward passes ending at
`Assessment:`—one asking the model to consider that the answer is factually correct and one asking
it to consider that it is factually incorrect. No assessment token is generated. The activation used
by HaloScope is the layer-wise difference `h_correct - h_incorrect`.

Run a fresh LLaMA-2 experiment:

```bash
haloscope all --config configs/prompt_contrast_llama2_7b_truthfulqa.yaml
```

OPT-6.7B is available without Meta's gated-model access:

```bash
haloscope all --config configs/prompt_contrast_opt_6.7b_truthfulqa.yaml
```

If the corresponding official baseline generations already exist, reuse the exact same answers and
extract only the prompted activations:

```bash
haloscope extract \
  --config configs/prompt_contrast_opt_6.7b_truthfulqa.yaml \
  --source-config configs/official_opt_6.7b_truthfulqa.yaml
haloscope label --config configs/prompt_contrast_opt_6.7b_truthfulqa.yaml
haloscope train --config configs/prompt_contrast_opt_6.7b_truthfulqa.yaml
haloscope evaluate --config configs/prompt_contrast_opt_6.7b_truthfulqa.yaml
```

The baseline and prompted results are written to their respective `metrics.json` files. Besides
`contrastive`, `model.activation_mode` accepts `response` (the existing behavior) and `diagnostic`
(one pre-verdict factuality prompt). Prompt templates support `{question}`, `{answer}`, `{context}`,
`{context_block}`, and `{prompt}` placeholders.

When scoring a new answer with a diagnostic or contrastive detector, pass the raw question so the
assessment prompt matches training:

```bash
haloscope score \
  --config configs/prompt_contrast_opt_6.7b_truthfulqa.yaml \
  --prompt "Answer the question concisely. Q: Who wrote Hamlet? A:" \
  --question "Who wrote Hamlet?" \
  --answer "William Shakespeare."
```

## Pre-generation and trajectory activation experiments

The additional activation modes leave every original mode and configuration unchanged:

- `prompt` extracts the state at the final token of the original prompt, before answer decoding.
- `endpoint_delta` uses `h_final - h_prompt`.
- `trajectory` concatenates `h_prompt`, changes from that state after configurable answer-token
  checkpoints, and `h_final - h_prompt`. With checkpoints `[1, 4, 8]`, its per-layer feature width
  is five times the model hidden size.

The supplied OPT trajectory profile writes to its own directory. Reuse the official run's exact
answers, then run the existing HaloScope labeling and detector stages:

```bash
haloscope extract \
  --config configs/trajectory_opt_6.7b_truthfulqa.yaml \
  --source-config configs/official_opt_6.7b_truthfulqa.yaml
haloscope label --config configs/trajectory_opt_6.7b_truthfulqa.yaml
haloscope train --config configs/trajectory_opt_6.7b_truthfulqa.yaml
haloscope evaluate --config configs/trajectory_opt_6.7b_truthfulqa.yaml
```

The same extraction can be submitted through Slurm:

```bash
sbatch scripts/slurm_haloscope.sbatch extract \
  configs/trajectory_opt_6.7b_truthfulqa.yaml \
  configs/official_opt_6.7b_truthfulqa.yaml
```

Trajectory extraction performs one prompt pass, one full-response pass, and one pass per configured
checkpoint. It is therefore slower than the final-token baseline but keeps peak memory similar when
`model.batch_size` is one. To run the prompt-only or endpoint-delta ablation, copy the trajectory
profile to a new filename and `work_dir`, then change `activation_mode`; never point two activation
experiments at the same work directory.

## Outputs

Each experiment writes:

```text
outputs/<experiment>/
├── examples.jsonl       normalized benchmark
├── generations.jsonl    prompts, answers, and references
├── embeddings.npy       [samples, layers, hidden_dim]
├── activation_metadata.json  extraction mode, templates, and source
├── labeled.jsonl        validation/evaluation similarity labels
├── split.npz            fixed seed-41 indices
├── detector/
│   ├── metadata.json
│   ├── subspace.npz
│   └── probe.pt or probe.pkl
└── metrics.json
```

`generate` is resumable. If a remote job stops, rerun the same command; it validates and continues
from the last complete checkpoint.

## Paper fidelity and deliberate options

- `orientation: paper` enforces the written assumption that high `ζ` means hallucination.
- `orientation: auto` checks both signs using validation labels, matching the authors' released code.
  The supplied full configs use `auto` for reference-code comparability.
- `load_in_4bit: false` preserves the original hidden states. Four-bit loading is supported on Linux
  as a memory-saving variant but is not an exact reproduction.
- The laptop profile uses a logistic classifier and ROUGE-L to stay small. The full profiles use the
  1,024-unit MLP and BLEURT.

Because the reference implementation and paper have minor operational differences, every choice is
saved in `detector/metadata.json`.
