#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/paper_exact_llama2_7b_truthfulqa.yaml}"
source .venv/bin/activate

# tee keeps a persistent log; each generation checkpoint is resumable.
mkdir -p logs
python -m haloscope.cli all --config "$CONFIG" 2>&1 | tee -a logs/haloscope.log

