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
- sklearn-compatible PCA component signs;
- the released raw-activation weighted projection score;
- 38 order-statistic pseudo-label thresholds;
- a 1,024-unit ReLU MLP trained for 50 epochs with SGD, momentum 0.9,
  weight decay 0.0003, and the released cosine schedule;
- final probe retraining after validation selection.

Use fresh output directories because older embeddings, labels, and splits were generated
under different choices:

```bash
sbatch scripts/slurm_haloscope.sbatch all configs/official_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch all configs/official_opt_6.7b_truthfulqa.yaml
```

Exact equality with a published AUROC is not guaranteed across GPU kernels, model or dataset
revisions, and nondeterministic CUDA operations. Report the exact commit, environment, and seed.
