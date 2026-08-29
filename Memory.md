# Project Memory & Agent Handoff State — Verixa

> **IMPORTANT NOTICE FOR FUTURE AI AGENTS:**
> This is a living, strictly factual handoff document. It reflects the **ACTUAL** verified state of the codebase, data, and models—not aspirational plans.
> - **Never** assume an experiment has been run; check actual outputs on disk.
> - **Never** overwrite or delete historical pilot artifacts.
> - **Never** silently alter seeds, splits, dataset sizes, or architectural constraints without explicit user approval.
> - Read [`Rules.md`](file:///c:/Verixa/Rules.md) and [`Phases.md`](file:///c:/Verixa/Phases.md) immediately after reading this document.

---

## 1. Project Overview

- **Project Purpose:** Verixa is an end-to-end deep learning framework designed to detect AI-generated synthetic imagery and maintain classification reliability under severe real-world transformations (JPEG compression, Gaussian blur, resizing, Gaussian noise, color jitter, and cropping).
- **Core Competition Constraints:**
  - **Hardware Budget:** NVIDIA GeForce RTX 4060 Laptop GPU, **8 GB VRAM** hard ceiling (peak allocated VRAM must stay $< 6.0\text{ GB}$).
  - **Model Complexity Limit:** Maximum **< 2 Billion parameters** (our models are ~28.6M to ~30.3M parameters, well within budget).
  - **Storage Ceiling:** Processed image storage target $< 500\text{ MB}$.

---

## 2. Current Development Status

- **Current Phase:** **Phase 4: FFT Experiment + Decision Checkpoint #2 — COMPLETED & ACCEPTED**
- **Completed Work:**
  - **Phase 1 (Dataset Pipeline):** Fully complete and committed (`30c8775`).
  - **Phase 2 Pilot Experiment (8K Images):** Completed, validated VRAM usage (< 550 MB), and permanently preserved as historical artifacts.
  - **Final 30K Dataset Preparation:** Ingestion of 30,000 images, deduplication, deterministic 80/20 train/val split, and cross-split leakage verification are **COMPLETE AND FROZEN**.
  - **Official 30K Phase 2 Baseline Training:** **COMPLETED AND VERIFIED**. Trained clean ConvNeXt-Tiny on the frozen 30K dataset (`data/manifests/merged_manifest.csv`). Best Epoch 2 achieved **AUROC 0.9964**, **Accuracy 97.03%**, and **FPR 2.80%**. Checkpoint saved to `models/convnext_tiny_baseline.pt` and report saved to `reports/baseline_clean_eval.json`.
  - **Phase 3 Robust RGB Training & Evaluation:** **COMPLETED AND VERIFIED**. Trained robust ConvNeXt-Tiny with on-the-fly transformation augmentations ($p=0.8$) on the identical frozen 30K dataset (`data/manifests/merged_manifest.csv`, seed `1337`). Best Epoch 3 achieved **AUROC 0.9964**, **Clean Accuracy 96.83%**, and **FPR 2.53%**. Checkpoint saved to `models/convnext_tiny_robust_rgb.pt`.
  - **Decision Checkpoint #1:** **APPROVED**. Designated `models/convnext_tiny_robust_rgb.pt` as the official **Locked Fallback Submission Model**.
  - **Phase 4 Standalone FFT Viability Experiment:** **COMPLETED**. Trained 1.63M parameter `FFTClassifier` from scratch on 2D Fourier magnitude spectra. Achieved **84.24% Clean Accuracy**, **94.65% AUROC**, and maintained **93.49% AUROC** under severe JPEG $Q=30$.
  - **Phase 4 Hybrid RGB + FFT Training & Evaluation:** **COMPLETED AND VERIFIED**. Trained dual-branch `HybridRGBFFTClassifier` (ConvNeXt-Tiny 768-d + 2D FFT CNN 512-d concatenation) on identical 30K dataset ($p=0.8$, seed `1337`). Best Epoch 6 achieved **97.55% Clean Accuracy**, **99.68% AUROC**, **96.96% Mean Transformed Accuracy**, and elevated worst-case floor to **95.40%**. Checkpoint saved to `models/convnext_tiny_hybrid_fft.pt`.
  - **Decision Checkpoint #2:** **ACCEPTED**. Designated `models/convnext_tiny_hybrid_fft.pt` as the **Current Best Candidate (Primary Submission Model)**; retained `models/convnext_tiny_robust_rgb.pt` as the **Locked Fallback Model**.
- **In Progress / Current Stage:**
  - Transition to Phase 5 (Held-Out Benchmark Evaluation).
- **Not Started:**
  - **Phase 5 (Held-Out Benchmark Evaluation):** Ready to begin.

---

## 3. Dataset State (Current Frozen 30K Dataset)

The training and validation dataset is **FROZEN** and stored in [`data/manifests/merged_manifest.csv`](file:///c:/Verixa/data/manifests/merged_manifest.csv).

### 3.1 Composition & Breakdown
- **CIFAKE:** **15,000 images** (7,500 Authentic `0` / 7,500 Synthetic `1`) sourced from local KaggleHub cache (`data/processed/cifake/`).
- **SID_Set:** **15,000 images** (7,500 Authentic `0` / 7,500 Full Synthetic `1`) streamed via Hugging Face `saberzl/SID_Set` (`data/processed/sid_set/`).
- **Total Dataset Size:** Exactly **30,000 images**.
- **Class Balance:** Perfectly balanced **50.0% REAL (15,000) / 50.0% AI-GENERATED (15,000)**.
- **Exclusion of SID_Set Tampered Images:** During streaming, **7,250 tampered images (`label=2`)** were detected and strictly excluded.
- **Exclusion of WildFake:** WildFake (`hy2628982280/WildFake` on ModelScope) is **strictly excluded from the training dataset**.

### 3.2 Standardization, Storage & Deterministic Split
- **Image Standardization:** All images are 3-channel RGB, bicubic resized to $224 \times 224$, compressed as JPEG ($Q=90$).
- **Actual On-Disk Storage:** **307.89 MB** (well within the 500 MB budget).
- **Deterministic Split Methodology:** Stratified 80/20 split assigned using global seed `1337` and lexicographic SHA-256 group sorting:
  - **Train Split (`train`):** **23,999 images** (80.0%)
  - **Validation Split (`val`):** **6,001 images** (20.0%)
- **Cross-Split Hash Leakage:** **Strictly 0**. ($\text{Train SHA} \cap \text{Val SHA} = \emptyset$).
- **Manifest Location:** [`data/manifests/merged_manifest.csv`](file:///c:/Verixa/data/manifests/merged_manifest.csv).

---

## 4. 8K Pilot History & Preserved Artifacts

- **Model Checkpoint:** [`models/convnext_tiny_baseline_8k_pilot.pt`](file:///c:/Verixa/models/convnext_tiny_baseline_8k_pilot.pt) (226.5 MB)
- **Evaluation Report:** [`reports/baseline_clean_eval_8k_pilot.json`](file:///c:/Verixa/reports/baseline_clean_eval_8k_pilot.json)
- **Pilot Metrics:** Best Epoch 3, AUROC 0.9935, Accuracy 96.50%, FPR 4.62%.

---

## 5. Model Benchmarks & Checkpoint Registry

### 5.1 Official Phase 2 Baseline (Clean Images Only)
- **Checkpoint:** [`models/convnext_tiny_baseline.pt`](file:///c:/Verixa/models/convnext_tiny_baseline.pt) (237.5 MB)
- **Report:** [`reports/baseline_clean_eval.json`](file:///c:/Verixa/reports/baseline_clean_eval.json) & [`reports/baseline_distortion_eval.json`](file:///c:/Verixa/reports/baseline_distortion_eval.json)
- **Best Epoch:** **Epoch 2** (Clean Acc: 97.03%, AUROC: 99.64%, FPR: 2.80%).

### 5.2 Official Phase 3 Robust Model (Locked Fallback Model)
- **Checkpoint:** [`models/convnext_tiny_robust_rgb.pt`](file:///c:/Verixa/models/convnext_tiny_robust_rgb.pt) (237.5 MB) — **LOCKED & PRESERVED**
- **Report:** [`reports/robust_rgb_clean_eval.json`](file:///c:/Verixa/reports/robust_rgb_clean_eval.json) & [`reports/robust_distortion_eval.json`](file:///c:/Verixa/reports/robust_distortion_eval.json)
- **Best Epoch:** **Epoch 3** (Clean Acc: 96.83%, AUROC: 99.64%, FPR: 2.53%).
- **Robustness:** Worst-case transformed accuracy: **94.33%** (vs 60.71% baseline). Severe noise accuracy: **94.53%** (+33.82 points).

### 5.3 Official Phase 4 Standalone FFT Model (Spectral Reference)
- **Checkpoint:** [`models/fft_standalone.pt`](file:///c:/Verixa/models/fft_standalone.pt) (13.2 MB)
- **Report:** [`reports/fft_standalone_clean_eval.json`](file:///c:/Verixa/reports/fft_standalone_clean_eval.json) & [`reports/fft_standalone_eval.json`](file:///c:/Verixa/reports/fft_standalone_eval.json)
- **Best Epoch:** **Epoch 8** (Clean Acc: 84.24%, AUROC: 94.65%, FPR: 5.43%).
- **Findings:** Confirmed frequency-domain AI generator artifacts exist; resilient to JPEG compression (AUROC 93.49% at Q=30), but vulnerable to severe Gaussian blur smoothing (AUROC 83.07% at $\sigma=2.0$).

### 5.4 Official Phase 4 Hybrid RGB + FFT Model (Current Best Candidate / Primary Submission Model)
- **Checkpoint:** [`models/convnext_tiny_hybrid_fft.pt`](file:///c:/Verixa/models/convnext_tiny_hybrid_fft.pt) (243.8 MB) — **PRIMARY SUBMISSION CANDIDATE**
- **Report:** [`reports/hybrid_fft_clean_eval.json`](file:///c:/Verixa/reports/hybrid_fft_clean_eval.json), [`reports/hybrid_distortion_eval.json`](file:///c:/Verixa/reports/hybrid_distortion_eval.json), and [`reports/fft_experiment_decision.json`](file:///c:/Verixa/reports/fft_experiment_decision.json)
- **Architecture:** ConvNeXt-Tiny spatial branch (768-d) + 2D FFT magnitude spectrum CNN (512-d) concatenated to 1,280-d projection head (~30.3M total params, 17.4M trainable).
- **Best Epoch:** **Epoch 6** (Clean Acc: **97.55%**, AUROC: **99.68%**, FPR: **2.50%**, FPR@95% TPR: **1.03%**).
- **Comparative Superiority Across All 16 Conditions:**
  - Outperformed Robust RGB fallback across **16 out of 16 conditions** (clean and all 15 distortions).
  - **Mean Transformed Accuracy:** **96.96%** (vs 96.11% for fallback, $+0.85\%$ gain).
  - **Mean Transformed AUROC:** **99.56%** (vs 99.40% for fallback).
  - **Worst-Case Robustness Floor:** Elevated from 94.33% to **95.40%** (under noise $\sigma=0.10$).
  - **Severe Distortions:** JPEG $Q=30$ reached **96.77%**; Blur $\sigma=2.0$ reached **96.68%**; Resize $0.25\times$ reached **96.27%**; Noise $\sigma=0.10$ reached **95.40%**.
  - **False Alarm Suppression:** FPR under severe blur dropped to **2.83%** (vs 3.80% fallback); FPR under severe resize dropped to **2.57%** (vs 3.93% fallback).

---

## 6. Existing Code & Repository Structure

| Path | Purpose |
| :--- | :--- |
| [`PRD.md`](file:///c:/Verixa/PRD.md) | Product requirements, 30K dataset structure, transformation specs, success criteria |
| [`Architecture.md`](file:///c:/Verixa/Architecture.md) | System dataflow, storage budgets, model design, hardware constraints |
| [`Rules.md`](file:///c:/Verixa/Rules.md) | Core development rules, scientific rigor, command execution policies |
| [`Phases.md`](file:///c:/Verixa/Phases.md) | Sequential 8-phase milestones and verifiable completion checklists |
| [`src/verixa/models/convnext.py`](file:///c:/Verixa/src/verixa/models/convnext.py) | ConvNeXt-Tiny binary classifier implementation and backbone stage freezing logic |
| [`src/verixa/models/fft.py`](file:///c:/Verixa/src/verixa/models/fft.py) | 2D FFT spectrum extractor (`FFTSpectrumExtractor`) and standalone spectral CNN (`FFTClassifier`) |
| [`src/verixa/models/hybrid.py`](file:///c:/Verixa/src/verixa/models/hybrid.py) | Dual-branch `HybridRGBFFTClassifier` and `freeze_hybrid_backbone_stages` |
| [`src/verixa/training/dataset.py`](file:///c:/Verixa/src/verixa/training/dataset.py) | PyTorch Dataset, custom transform support, and eval DataLoader factories |
| [`src/verixa/training/augmentations.py`](file:///c:/Verixa/src/verixa/training/augmentations.py) | CPU-side 6-family augmentation engine and pickle-safe discrete distortion suites |
| [`src/verixa/training/trainer.py`](file:///c:/Verixa/src/verixa/training/trainer.py) | AMP training loop, in-place ASCII progress display, VRAM tracker, early stopping |
| [`src/verixa/evaluation/metrics.py`](file:///c:/Verixa/src/verixa/evaluation/metrics.py) | Evaluation engine: Accuracy, AUROC, FPR at threshold, FPR at 95% TPR, confusion matrix |
| [`scripts/train_rgb.py`](file:///c:/Verixa/scripts/train_rgb.py) | CLI training script for clean baseline and robust RGB models |
| [`scripts/train_fft_standalone.py`](file:///c:/Verixa/scripts/train_fft_standalone.py) | CLI training script for standalone FFT-only model |
| [`scripts/train_hybrid.py`](file:///c:/Verixa/scripts/train_hybrid.py) | CLI training script for Hybrid RGB+FFT dual-branch model |
| [`scripts/evaluate_robustness.py`](file:///c:/Verixa/scripts/evaluate_robustness.py) | CLI evaluation harness supporting any model architecture across 16 distortion conditions |
| [`tests/test_hybrid.py`](file:///c:/Verixa/tests/test_hybrid.py) | Unit tests for Hybrid RGB+FFT model, forward passes, and freezing |
| [`tests/test_fft.py`](file:///c:/Verixa/tests/test_fft.py) | Unit tests for FFT spectrum extractor and classifier |
| [`tests/test_augmentations.py`](file:///c:/Verixa/tests/test_augmentations.py) | Unit tests for all 6 augmentation families and evaluation suites |
| [`tests/test_convnext.py`](file:///c:/Verixa/tests/test_convnext.py) | Unit tests for ConvNeXt forward pass, freezing, and binary metrics |
| [`tests/test_manifest.py`](file:///c:/Verixa/tests/test_manifest.py) | Unit tests for deterministic splitting and manifest statistics |
| [`tests/test_dedupe.py`](file:///c:/Verixa/tests/test_dedupe.py) | Unit tests for hash deduplication and split leakage detection |
| [`tests/test_schema.py`](file:///c:/Verixa/tests/test_schema.py) | Unit tests for label parsing and validation |

---

## 7. Verification Status

- **Automated Tests:** **56/56 passed** (`pytest -v`).
- **Linter & Formatting:** **All checks passed** (`ruff check .`).
- **Compute Environment:** Python 3.11.16, PyTorch 2.6.0+cu128, CUDA 12.8 active on Windows.
- **GPU Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,192 MB VRAM, measured hybrid peak ~870 MB).

---

## 8. Critical Decisions Already Made (Do NOT Silently Reverse)

1. **Hybrid RGB + FFT is the Primary Candidate:** Selected based on superior performance across all 16 evaluation conditions (**97.55% clean accuracy**, **96.96% mean transformed accuracy**, **95.40% worst-case floor**).
2. **Robust RGB Model is the Locked Fallback:** [`models/convnext_tiny_robust_rgb.pt`](file:///c:/Verixa/models/convnext_tiny_robust_rgb.pt) is permanently preserved as the official fallback submission model and must never be modified.
3. **No Further Architecture Iterations:** The architecture is frozen. No cross-attention, gating, or additional branches.
4. **No Training From Scratch for Backbone:** ConvNeXt-Tiny uses ImageNet-1K pretrained weights; Stage 3 + head are fine-tuned.
5. **No Ensembles or Giant Models:** The entire hybrid is ~30.3M parameters, $< 1.6\%$ of the 2B parameter limit.
6. **Held-Out Benchmark Isolation:** `data/held_out_benchmark/` has remained completely untouched and unread throughout training and validation. It is reserved strictly for Phase 5 single-pass evaluation.
7. **Strict Head-to-Head Experimental Control:** All models were trained on the **identical frozen 30K dataset and split** (`data/manifests/merged_manifest.csv`, seed `1337`) with identical augmentation settings ($p=0.8$) where applicable.

---

## 9. AI-Agent Workflow & User Collaboration Rules

- **User Runs Commands:** The user personally executes evaluation commands locally in their VS Code PowerShell terminal to view live progress.
- **Direct Python Binary:** Always provide commands using direct Python binary (`& 'C:\Users\amite\anaconda3\envs\verixa\python.exe' ...`).
- **ASCII-Only Display:** Use 7-bit ASCII characters (`#` and `-`) for progress bars to avoid Windows PowerShell encoding issues.
- **In-Place Progress Updates:** Use single-line `\r` updates with permanent summary prints upon stage completion.

---

## 10. Git State

- **Completed Commits:**
  - `30c8775` — `feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training and evaluation`
  - `c5062b9` — `feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training on 30K dataset`
  - `6b0e3f8` — `feat(training): complete Phase 3 transformation-aware training and Checkpoint 1`
- **Current Working Tree:** All Phase 4 deliverables complete and verified. Ready to commit:
  `feat(exp): complete Phase 4 FFT dual-branch experiment and Checkpoint 2`

---

## 11. NEXT ACTION

1. Commit Phase 4 deliverables:
   `git commit -m "feat(exp): complete Phase 4 FFT dual-branch experiment and Checkpoint 2"`
2. Verify `git status` is 100% clean.
3. Proceed to **Phase 5: Final Robustness Evaluation & Held-Out Benchmark**:
   - Verify held-out benchmark integrity (`data/held_out_benchmark/`).
   - Run full 16-condition evaluation report generation for documentation.
   - Run single-pass final evaluation on the isolated held-out benchmark for both the Primary Model (`models/convnext_tiny_hybrid_fft.pt`) and Fallback Model (`models/convnext_tiny_robust_rgb.pt`).

---

## 12. Incoming Agent Checklist

When you resume or take over this repository:
- [ ] Read `Memory.md` first.
- [ ] Read `Rules.md` for strict behavioral guidelines.
- [ ] Read `Phases.md` for milestone requirements.
- [ ] Run `git status` to verify the exact state of the working tree.
- [ ] Run `pytest -v` and `ruff check .` before modifying any code.
- [ ] Never modify or overwrite locked checkpoints on disk.
- [ ] Follow the exact Next Action outlined in Section 11.
