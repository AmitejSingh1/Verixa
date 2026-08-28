# Verixa Project Explanation

This file is the running plain-English record of what changed, why it changed, and what to do next. Keep it updated at the end of every phase.

## Phase 1: Dataset Inspection, Pilot Pipeline, And Scaling

Status: in progress

What has been set up:

- Created the Python project skeleton under `src/verixa`, with scripts under `scripts`.
- Created runtime folders for `data`, `models`, `eval`, `experiments`, and `reports`.
- Added a CUDA-first Conda setup using your existing Anaconda install at `C:\Users\amite\anaconda3`.
- Created the `verixa` Conda environment with Python 3.11.
- Installed CUDA-enabled PyTorch inside that env.
- Verified PyTorch can use the local `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Added lightweight tests and linting so early project changes stay clean.

Why this matters:

- The challenge rewards robustness under transformations, so the project needs a repeatable data pipeline and fixed validation split before model tuning.
- The RTX 4060 has 8 GB VRAM, so every later training script must explicitly log GPU usage and avoid silent CPU fallback.
- Dataset labels need to be standardized before training because the sources do not all expose the same schema.

Dataset findings so far:

- SID_Set has `train` and `validation` splits.
- SID_Set sample labels are numeric: `0`, `1`, and `2`.
- Sample image IDs indicate `1` corresponds to `full_synthetic` and `2` corresponds to `tampered`.
- Recommended SID_Set binary mapping for this detector: use `0 -> real`, `1 -> AI-generated`, and exclude `2 -> tampered` from core binary training unless intentionally added later as a separate experiment.
- WildFake inspection is currently opt-in because the ModelScope translation/access step can hang or need manual setup.

Current verification:

- `python -m compileall src scripts` passes in the `verixa` Conda env.
- `python -m pytest` passes.
- `python -m ruff check .` passes.
- `python scripts/check_cuda.py` confirms CUDA is available.

Next Phase 1 work:

- Complete CIFAKE metadata inspection.
- Add or run source-specific ingestion for SID_Set and CIFAKE.
- Keep WildFake as a generator-diversity source once ModelScope access is reliable.
- Build a ~10K pilot manifest, run duplicate/leakage checks, create the fixed validation split, and produce the Phase 1 dataset statistics report.
