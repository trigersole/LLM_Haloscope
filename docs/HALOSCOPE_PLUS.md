# HaloScope++ experiments

This branch keeps the original configurations backward compatible. The improved
experiments write to new output directories and never overwrite the baseline.

## Phase A: improve training without running Llama again

Confirm that `artifact_source_dir` in
`configs/haloscope_plus_phase_a_llama2_7b_truthfulqa.yaml` points to the output
directory containing the baseline `embeddings.npy`, `labeled.jsonl`, and
`split.npz`. Then run:

```bash
python -m haloscope.cli train \
  --config configs/haloscope_plus_phase_a_llama2_7b_truthfulqa.yaml
python -m haloscope.cli evaluate \
  --config configs/haloscope_plus_phase_a_llama2_7b_truthfulqa.yaml
```

This reuses all baseline evaluation artifacts without copying the large activation
file. It changes only PCA scoring, pseudo-label selection, and probe training.
The supplied configuration uses a 128-unit dropout MLP and averages three seeds.
For the deterministic linear baseline, change `backend` to `logistic`, set
`probe_repeats` to `1`, and leave the remaining search settings unchanged.

On Slurm, submit training first and make evaluation depend on its success:

```bash
TRAIN_JOB=$(sbatch --parsable scripts/slurm_haloscope.sbatch \
  train configs/haloscope_plus_phase_a_llama2_7b_truthfulqa.yaml)
sbatch --dependency="afterok:${TRAIN_JOB}" scripts/slurm_haloscope.sbatch \
  evaluate configs/haloscope_plus_phase_a_llama2_7b_truthfulqa.yaml
```

## Phase B: endpoint-delta representation

Reuse the baseline answers but perform a new teacher-forced activation pass:

```bash
python -m haloscope.cli extract \
  --config configs/haloscope_plus_endpoint_delta_llama2_7b_truthfulqa.yaml \
  --source-config configs/official_llama2_7b_truthfulqa.yaml
python -m haloscope.cli train \
  --config configs/haloscope_plus_endpoint_delta_llama2_7b_truthfulqa.yaml
python -m haloscope.cli evaluate \
  --config configs/haloscope_plus_endpoint_delta_llama2_7b_truthfulqa.yaml
```

`extract` copies the small generation, label, and split artifacts, then writes new
endpoint-delta embeddings to a separate output directory. It does not generate new
answers or rerun BLEURT.

The equivalent Slurm sequence is:

```bash
EXTRACT_JOB=$(sbatch --parsable scripts/slurm_haloscope.sbatch \
  extract configs/haloscope_plus_endpoint_delta_llama2_7b_truthfulqa.yaml \
  configs/official_llama2_7b_truthfulqa.yaml)
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:${EXTRACT_JOB}" \
  scripts/slurm_haloscope.sbatch train \
  configs/haloscope_plus_endpoint_delta_llama2_7b_truthfulqa.yaml)
sbatch --dependency="afterok:${TRAIN_JOB}" scripts/slurm_haloscope.sbatch \
  evaluate configs/haloscope_plus_endpoint_delta_llama2_7b_truthfulqa.yaml
```

## New configuration options

- `pseudo_label_mode: confidence_tails` keeps only both score extremes.
- `tail_fractions` searches the fraction retained from each tail.
- `confidence_weighted` gives more extreme examples greater training weight.
- `probe_repeats` trains deterministic seeds and averages their probabilities.
- `validation_folds` selects candidates by mean stratified fold AUROC.
- `stability_penalty` subtracts a multiple of fold AUROC standard deviation.
- `probe.dropout`, `probe.penalty`, `probe.balanced_loss`, `probe.optimizer`, and
  `probe.standardize` configure the probe.

Existing configs omit these fields and preserve their previous behavior.
