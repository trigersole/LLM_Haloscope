# Running the full experiment through PuTTY

PuTTY is only the SSH client. The model runs on a Linux machine with an NVIDIA GPU; closing
PuTTY must not terminate the job, so run it inside `tmux`.

## Recommended remote machine

- Ubuntu 22.04/24.04
- Python 3.10 or 3.11
- NVIDIA driver with a CUDA-capable PyTorch build
- 24 GB VRAM for unquantized LLaMA-2-7B or OPT-6.7B; 40 GB gives more headroom
- Roughly 50 GB free disk for model weights, caches, activations, and checkpoints

The full paper profile intentionally does not use 4-bit quantization because quantization changes
the hidden states being measured. If only a 12–16 GB GPU is available, set `load_in_4bit: true`,
install `.[quantization]`, and treat the result as a practical variant rather than an exact reproduction.

## 1. Copy the project from Windows

Install PuTTY (which includes `pscp.exe`), open PowerShell in the directory containing this project,
and run:

```powershell
pscp.exe -r .\Haloscope username@SERVER_IP:/home/username/
```

Use a private key with `-i C:\path\key.ppk` when password login is disabled.

## 2. Connect and prepare

Open PuTTY, enter the server IP, log in, then:

```bash
cd ~/Haloscope
bash scripts/setup_remote.sh
nvidia-smi
```

For gated LLaMA-2, accept Meta's model license on Hugging Face and authenticate:

```bash
source .venv/bin/activate
huggingface-cli login
```

OPT-6.7B is not gated and can be selected with
`configs/paper_exact_opt_6.7b_truthfulqa.yaml`.

## 3. Start a persistent run

```bash
tmux new -s haloscope
cd ~/Haloscope
bash scripts/run_remote.sh configs/paper_exact_llama2_7b_truthfulqa.yaml
```

Detach with `Ctrl+B`, then `D`. PuTTY can now be closed. Reconnect later with:

```bash
tmux attach -t haloscope
```

The log is also available at `logs/haloscope.log`. GPU usage can be checked in a second PuTTY
session with `watch -n 2 nvidia-smi`.

## 4. Resume after an interruption

Run the same command again. Generation checkpoints are written atomically every ten samples, so
completed samples are skipped. The later stages can also be run separately:

```bash
source .venv/bin/activate
haloscope generate --config configs/paper_exact_llama2_7b_truthfulqa.yaml
haloscope label --config configs/paper_exact_llama2_7b_truthfulqa.yaml
haloscope train --config configs/paper_exact_llama2_7b_truthfulqa.yaml
haloscope evaluate --config configs/paper_exact_llama2_7b_truthfulqa.yaml
```

## 5. Download the results

From Windows PowerShell:
```powershell
```
pscp.exe -r username@SERVER_IP:/home/username/Haloscope/outputs/paper_llama2_7b_truthfulqa .\results\


The key outputs are `metrics.json`, `detector/metadata.json`, `detector/subspace.npz`, and the
trained probe file.

