#!/usr/bin/env bash
#SBATCH --job-name=haloscope
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/slurm_haloscope.sbatch all configs/paper_exact_llama2_7b_truthfulqa.yaml
# The first argument may also be prepare, generate, label, train, or evaluate.
STAGE="${1:-all}"
CONFIG="${2:-configs/paper_exact_llama2_7b_truthfulqa.yaml}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$PROJECT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Missing .venv. Run: bash scripts/setup_remote.sh"
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration does not exist: $CONFIG"
    exit 2
fi

source .venv/bin/activate

# Keep downloads and experiment state on persistent storage, not node-local scratch.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$HF_HOME" logs

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Stage: $STAGE"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi || true
python -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU visible")'

# Slurm sends TERM two minutes before the wall-time limit. Previously completed generation
# checkpoints remain valid; submit the same command again to continue.
trap 'echo "Received termination signal; resubmit the same job to resume from checkpoints."' TERM

srun --ntasks=1 python -u -m haloscope.cli "$STAGE" --config "$CONFIG"

