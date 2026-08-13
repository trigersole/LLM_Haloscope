$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[llm]"
python -m haloscope.cli smoke
python -m haloscope.cli all --config configs/laptop_smollm2.yaml

