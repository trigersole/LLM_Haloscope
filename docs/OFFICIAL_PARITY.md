# Official-code parity

The `official_*` profiles reproduce the operational choices in
`deeplearning-wisc/haloscope`, while retaining this project's resumable files and CLI.

Implemented parity points:

- seed 41 with NumPy's legacy MT19937 permutation;
- partitions rebuilt in original dataset order after membership selection;
- prompt `Answer the question concisely. Q: {question} A:`;
- five-beam deterministic generation with 64 new tokens;
- eager attention, matching the explicit attention computation in released `llama_iti`;
- released TruthfulQA cleanup of repeated `Answer the question concisely` text,
  including preservation of the decoded answer's trailing whitespace;
- final-token transformer-block representations (`feat_loc_svd=3`);
- BLEURT threshold 0.5 and the released candidate/reference input order;
- validation-fitted PCA for selecting layer, component count, and sign;
- float32 sklearn PCA with its released `svd_solver="auto"` behavior;
- replay of the legacy split permutation before randomized PCA consumes NumPy RNG;
- the released raw-activation weighted projection score;
- 38 order-statistic pseudo-label thresholds;
- a 1,024-unit ReLU MLP trained for 50 epochs with SGD, momentum 0.9,
  weight decay 0.0003, and the released cosine schedule;
- final probe retraining after validation selection.
- released class-grouped ordering of pseudo-labeled probe inputs.

Compare a completed run with artifacts produced by the authors' repository:

```bash
python scripts/audit_official_parity.py \
  --ours outputs/official_parity_v2_llama2_7b_truthfulqa \
  --official /home/msai/$USER/haloscope-official \
  --output outputs/official_parity_v2_llama2_7b_truthfulqa/parity_audit.json
```

The audit checks exact generated answers, BLEURT scores and threshold labels, split
membership, and every aligned transformer-block embedding layer. BLEURT and embedding
differences are also reported separately for identical-answer and different-answer subsets.
Run it before comparing selected hyperparameters or final AUROC; the first divergent upstream
artifact explains all downstream differences.

To reuse an isolated official dependency overlay on Slurm without changing this project's
virtual environment:

```bash
sbatch \
  --export=ALL,HALOSCOPE_DEPENDENCY_OVERLAY=/home/msai/$USER/haloscope-official/.deps \
  scripts/slurm_haloscope.sbatch \
  train configs/official_llama2_7b_truthfulqa.yaml
```

For an `all` job, the launcher deliberately runs dataset preparation before enabling the
overlay. This prevents an older overlay copy of `datasets` from reading cache metadata written
by a newer release; generation, labeling, training, and evaluation then use the overlay.

Use a fresh `work_dir` because older answers and embeddings were saved before the released
answer cleanup was implemented. The supplied v2 profiles already point to new directories;
do not change them back to an old generation checkpoint.

```bash
sbatch scripts/slurm_haloscope.sbatch all configs/official_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch all configs/official_opt_6.7b_truthfulqa.yaml
```

Exact equality with a published AUROC is not guaranteed across GPU kernels, model or dataset
revisions, and nondeterministic CUDA operations. Report the exact commit, environment, and seed.
