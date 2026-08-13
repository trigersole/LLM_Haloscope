# Running HaloScope with Slurm

The supplied batch job uses one GPU, one task, eight CPU cores, 64 GB host RAM, and a 48-hour
limit. Change these values to match the policies and GPU names on your cluster.

## 1. Copy and install

On the cluster login node:

```bash
cd ~/Haloscope

# Load your site's Python/CUDA modules first if required. Names differ by cluster, for example:
# module load python/3.11 cuda/12.4

bash scripts/setup_remote.sh
```

LLaMA-2 is gated. Accept its Hugging Face license and authenticate before submitting:

```bash
source .venv/bin/activate
huggingface-cli login
```

Use the OPT configuration if gated LLaMA access is unavailable.

## 2. Submit the complete experiment

The log directory must exist before `sbatch` opens its output file:

```bash
mkdir -p logs
sbatch scripts/slurm_haloscope.sbatch \
  all configs/paper_exact_llama2_7b_truthfulqa.yaml
```

The submission prints a job ID such as `Submitted batch job 12345`.

If your site requires a partition and account, supply them at submission time:

```bash
sbatch \
  --partition=gpu \
  --account=YOUR_ACCOUNT \
  --gres=gpu:a100:1 \
  --time=2-00:00:00 \
  scripts/slurm_haloscope.sbatch \
  all configs/paper_exact_llama2_7b_truthfulqa.yaml
```

GPU resource syntax is site-dependent. Some clusters use `--gpus=1` instead of
`--gres=gpu:1`; follow the example from your cluster's documentation.

## 3. Monitor it

```bash
squeue -u "$USER"
tail -f logs/slurm-haloscope-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Cancel a job with:

```bash
scancel JOB_ID
```

## 4. Resume a timed-out or interrupted job

Submit exactly the same command again. The generation stage validates `generations.jsonl` and
`embeddings.npy` and continues after the last atomic checkpoint:

```bash
sbatch scripts/slurm_haloscope.sbatch \
  all configs/paper_exact_llama2_7b_truthfulqa.yaml
```

Up to the currently active ten-sample block may be repeated after termination; completed blocks are
not regenerated.

## 5. Run individual stages

The first argument is the stage and the second is the configuration:

```bash
sbatch scripts/slurm_haloscope.sbatch prepare  configs/paper_exact_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch generate configs/paper_exact_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch label    configs/paper_exact_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch train    configs/paper_exact_llama2_7b_truthfulqa.yaml
sbatch scripts/slurm_haloscope.sbatch evaluate configs/paper_exact_llama2_7b_truthfulqa.yaml
```

Wait for each stage to finish before submitting the next one. Normally the single `all` job is easier
and prevents accidental ordering mistakes.

## Common cluster problems

- **`Requested node configuration is not available`:** use the cluster's actual GPU partition,
  account, GPU type, memory limit, and wall-time limit.
- **PyTorch says CUDA is unavailable:** load the correct CUDA/Python module before creating the
  virtual environment, or install the CUDA-enabled PyTorch wheel recommended by the cluster.
- **CUDA out of memory:** request a GPU with at least 24 GB VRAM, reduce `num_beams`, or set
  `load_in_4bit: true`. Quantization is a practical variant, not an exact paper reproduction.
- **No space in the home directory:** set `HF_HOME` to persistent project/scratch storage when
  submitting, for example `sbatch --export=ALL,HF_HOME=/scratch/$USER/huggingface ...`.
- **LLaMA returns 401/403:** authenticate on the login node and confirm the model license was
  accepted. The token cache must be visible from compute nodes.

