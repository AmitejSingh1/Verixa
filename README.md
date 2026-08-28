# Verixa

Verixa is a hackathon prototype for robust AI-generated image detection under real-world image transformations. The primary model path is a CUDA-accelerated PyTorch `ConvNeXt-Tiny` binary classifier trained on resized `224x224` JPEGs with a fixed, leakage-checked validation split.

## Current Status

This repository is being built phase by phase.

- Phase 1: dataset inspection, pilot ingestion, manifest creation, duplicate checks, and fixed validation split
- Phase 2: clean RGB ConvNeXt-Tiny baseline
- Phase 3: transformation-aware robust RGB training and evaluation
- Phase 4: optional FFT experiment behind a go/no-go checkpoint
- Phase 5-8: final benchmark, error analysis, inference script, and documentation polish

## CUDA Conda Setup

Use a fresh Anaconda environment so CUDA/PyTorch dependencies do not collide with older projects.

Open Anaconda Prompt or a PowerShell session where `conda` is available:

```powershell
conda create -n verixa python=3.11 pip -y
conda activate verixa
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/check_cuda.py
```

On this machine, Conda is available at `C:\Users\amite\anaconda3\Scripts\conda.exe`. If regular PowerShell does not know `conda`, use:

```powershell
& 'C:\Users\amite\anaconda3\Scripts\conda.exe' create -n verixa python=3.11 pip -y
& 'C:\Users\amite\anaconda3\Scripts\conda.exe' run -n verixa python -m pip install -r requirements-torch-cu128.txt
& 'C:\Users\amite\anaconda3\Scripts\conda.exe' run -n verixa python -m pip install -r requirements.txt
& 'C:\Users\amite\anaconda3\Scripts\conda.exe' run -n verixa python -m pip install -r requirements-dev.txt
& 'C:\Users\amite\anaconda3\Scripts\conda.exe' run -n verixa python -m pip install -e .
& 'C:\Users\amite\anaconda3\Scripts\conda.exe' run -n verixa python scripts/check_cuda.py
```

This project intentionally installs CUDA PyTorch from the official PyTorch wheel index inside the Conda environment. Conda still provides the isolated Python environment.

If `conda` is not available in your current PowerShell, launch Anaconda Prompt or initialize Conda for PowerShell from your Anaconda install.

## Phase 1 Workflow

Start with metadata inspection before downloading or sampling images:

```powershell
python scripts/inspect_datasets.py --out reports/dataset_inspection.json
```

WildFake inspection is opt-in because the ModelScope translation/access step may need manual setup first:

```powershell
python scripts/inspect_datasets.py --include-wildfake --out reports/dataset_inspection.json
```

SID_Set label mapping is a required manual checkpoint. If the inspection report shows ambiguous categories such as tampered, edited, manipulated, or synthetic variants that are not clearly AI-generated, stop and decide the binary mapping before ingestion or training.

For local folder-style datasets such as CIFAKE, use:

```powershell
python scripts/inspect_local_dataset.py --root path\to\dataset --out reports/cifake_local_inspection.json
```

Then ingest only after labels are mapped:

```powershell
python scripts/ingest_local_dataset.py `
  --root path\to\dataset `
  --source-dataset CIFAKE `
  --label-map config\cifake_label_map.json `
  --output-root data\processed\cifake `
  --manifest data\manifests\cifake_manifest.csv `
  --limit-per-class 5000
```

All ingested images are resized to `224x224` and saved as compressed JPEG. Full-resolution originals should not be copied into this repository.

## Verification

Run the lightweight checks:

```powershell
python -m compileall src scripts
python -m pytest
```

## Notes

- Do not touch the held-out benchmark until Phase 5.
- Keep one fixed validation split for all comparisons.
- Log peak GPU VRAM for every training run.
- If VRAM approaches the 8 GB laptop limit, pause and record the measured usage before changing the experiment.
