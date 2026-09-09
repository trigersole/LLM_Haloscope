# Official-code parity

The `official_*` profiles reproduce the operational choices in
`deeplearning-wisc/haloscope`, while retaining this project's resumable files and CLI.

Implemented parity points:

- seed 41 with NumPy's legacy MT19937 permutation;
- partitions rebuilt in original dataset order after membership selection;
- prompt `Answer the question concisely. Q: {question} A:`;
- five-beam deterministic generation with 64 new tokens;
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
  --ours outputs/official_llama2_7b_truthfulqa \
  --official /home/msai/$USER/haloscope-official \
  --output outputs/official_llama2_7b_truthfulqa/parity_audit.json
```

The audit checks exact generated answers, BLEURT scores and threshold labels, split
membership, and every aligned transformer-block embedding layer. Run it before comparing
selected hyperparameters or final AUROC; the first divergent upstream artifact explains all
downstream differences.

To reuse an isolated official dependency overlay on Slurm without changing this project's
virtual environment:

```bash
sbatch \
  --export=ALL,HALOSCOPE_DEPENDENCY_OVERLAY=/home/msai/$USER/haloscope-official/.deps \
  scripts/slurm_haloscope.sbatch \
  train configs/official_llama2_7b_truthfulqa.yaml
```

Use fresh output directories because older embeddings, labels, and splits were generated
under different choices:

```bash
sbatch scripts/slurm_haloscope.sbatch all configs/official_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch all configs/official_opt_6.7b_truthfulqa.yaml
```

Exact equality with a published AUROC is not guaranteed across GPU kernels, model or dataset
revisions, and nondeterministic CUDA operations. Report the exact commit, environment, and seed.
