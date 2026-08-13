#!/usr/bin/env bash
#SBATCH --job-name=haloscope-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/slurm_haloscope.sbatch
#   sbatch scripts/slurm_haloscope.sbatch all configs/paper_exact_opt_6.7b_truthfulqa.yaml
# The first argument may also be prepare, generate, label, train, or evaluate.
STAGE="${1:-all}"
CONFIG="${2:-configs/paper_exact_opt_6.7b_truthfulqa.yaml}"
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
python -c 'import sys, torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA build:", torch.version.cuda); print("GPU count:", torch.cuda.device_count()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); sys.exit(0 if torch.cuda.is_available() else 1)'

# Disk usage is relatively expensive to calculate, so record it only at job start and exit.
print_disk_usage() {
    echo "[$(date --iso-8601=seconds)] Disk usage"
    df -h "$PROJECT_DIR" || true
    du -sh "$PROJECT_DIR" "$HF_HOME" 2>/dev/null || true
    if [[ -n "${SCRATCH:-}" ]]; then
        df -h "$SCRATCH" || true
    fi
    quota -s 2>/dev/null || true
}

# These snapshots are cheap and run once per minute. `free` is node-wide; `sstat` reports
# the Slurm job/step usage. GPU values are for the GPU assigned via CUDA_VISIBLE_DEVICES.
print_runtime_usage() {
    echo "[$(date --iso-8601=seconds)] Runtime resources"
    free -h || true
    nvidia-smi \
      --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader || true
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        sstat -j "$SLURM_JOB_ID" \
          --noheader \
          --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize 2>/dev/null || true
    fi
}

monitor_resources() {
    while true; do
        print_runtime_usage
        sleep 60
    done
}

MONITOR_PID=""
cleanup() {
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    echo "Final resource snapshot"
    print_runtime_usage
    print_disk_usage
}

on_term() {
    echo "Received termination signal; resubmit the same job to resume from checkpoints."
    exit 143
}

print_disk_usage
print_runtime_usage
monitor_resources &
MONITOR_PID=$!

# Slurm sends TERM two minutes before the wall-time limit. Previously completed generation
# checkpoints remain valid; submit the same command again to continue.
trap cleanup EXIT
trap on_term TERM

srun --ntasks=1 python -u -m haloscope.cli "$STAGE" --config "$CONFIG"
