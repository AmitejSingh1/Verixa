# Project Memory & Agent Handoff State — Verixa

> **CRITICAL ARCHITECTURAL DIRECTIVE FOR FUTURE AGENTS:**
> - The model-selection phase is **PERMANENTLY CLOSED**.
> - **Final Champion Model:** [`models/convnext_tiny_hybrid_fft.pt`](file:///c:/Verixa/models/convnext_tiny_hybrid_fft.pt) (Hybrid V1).
> - **Locked Fallback Model:** [`models/convnext_tiny_robust_rgb.pt`](file:///c:/Verixa/models/convnext_tiny_robust_rgb.pt).
> - **Deployment Threshold:** **PERMANENTLY LOCKED AT 0.50**.
> - **Hybrid V4 is REJECTED.** Do **NOT** train any more models.
> - **NEVER** modify or overwrite the champion or fallback checkpoints.
> - **STRICT HELD-OUT QUARANTINE:** `benchmark_dataset/` was evaluated once under quarantine and must remain completely untouched until the final single-pass evaluation.

---

## 1. Project Overview & Operational Envelope

- **Project Purpose:** Verixa is an end-to-end deep learning framework designed to detect AI-generated synthetic imagery and maintain classification reliability under severe real-world transformations (JPEG compression, Gaussian blur, resizing, Gaussian noise, color jitter, and cropping).
- **Core Competition Constraints:**
  - **Hardware Budget:** NVIDIA GeForce RTX 4060 Laptop GPU, **8 GB VRAM** ceiling (actual peak training VRAM $< 1.3\text{ GB}$, inference $< 400\text{ MB}$).
  - **Model Complexity Limit:** Hard limit **< 2 Billion parameters** (our hybrid model is $29,784,130$ total parameters / 17.4M trainable, $< 1.5\%$ of the budget).
  - **Storage Ceiling:** Processed image storage target $< 500\text{ MB}$ (actual on-disk storage is **307.89 MB** for 30,000 images).

---

## 2. Current Development Status — ALL 8 PHASES COMPLETE

1. **Phase 1 (Dataset Pipeline):** Ingested 30,000 images (15,000 CIFAKE + 15,000 SID_Set), deduplicated, assigned fixed stratified split (23,999 train / 6,001 val, seed `1337`), strictly zero cross-split leakage.
2. **Phase 2 (Clean RGB ConvNeXt Baseline):** Clean Acc 97.03%, AUROC 99.64%, FPR 2.80%. Suffered catastrophic collapse under severe noise ($\sigma=0.10$ fell to 60.71%). Checkpoint preserved as `models/convnext_tiny_baseline.pt`.
3. **Phase 3 (Transformation-Aware Robust RGB):** Trained with $p=0.8$ dynamic augmentations. Clean Acc 96.83%, AUROC 99.64%, FPR 2.53%. Noise robustness floor elevated to 94.53% (+33.82 points). Approved as **Locked Fallback Model** (`models/convnext_tiny_robust_rgb.pt`).
4. **Phase 4 (FFT & Hybrid Fusion Architecture):** Standalone FFT achieved 84.24% clean acc and 94.65% AUROC. Hybrid RGB+FFT V1 achieved 97.55% clean acc, 99.68% AUROC, 96.95% mean transformed acc, and 95.37% worst-case floor. Approved as **Primary Champion Model** (`models/convnext_tiny_hybrid_fft.pt`).
5. **Phase 5 (Held-Out Benchmark Evaluation & Post-Phase 5 Ablations):**
   - Held-Out Benchmark ($N=13,841$: 4,998 COCO val2017 authentic + 8,843 DALL-E 3 synthetic):
     - Hybrid V1: **92.13% AUROC**, **68.34% Overall Acc**, **97.30% COCO Acc** (**2.70% FPR**), **51.97% DALL-E 3 Recall**.
     - Robust RGB Fallback: 90.74% AUROC, 66.79% Overall Acc, 94.36% COCO Acc (5.64% FPR), 51.20% DALL-E 3 Recall.
     - Hybrid reduced false positive alarms on authentic photos by **52.1%** ($5.64\% \rightarrow 2.70\%$).
   - Canonical Threshold Calibration: Recomputed fresh predictions across 6,001 development validation images on GPU; confirmed deployment threshold $\theta = 0.50$ provides optimal worst-case robustness floor (95.55%) with clean FPR $\le 2.50\%$. Report locked in `reports/threshold_calibration.json`.
   - Hybrid V2 (unfreeze Stage 2): Degraded noise floor to 93.75%. REJECTED.
   - Hybrid V3 (clean ablation: label smoothing + flip): Degraded clean acc to 96.80% and mean transformed acc to 96.41%. REJECTED.
   - Hybrid V4 (`pos_weight = 1.35`): Authoritative actual results: Clean Acc **96.53%**, Clean FPR **2.07%**, Mean Transformed Acc **96.03%**, Worst-Case Floor **94.75%**. Failed promotion gates 1 and 3. REJECTED.
6. **Phase 6 (Comprehensive Forensic Error Analysis):** Completed in [`reports/error_analysis.md`](file:///c:/Verixa/reports/error_analysis.md). Analyzed COCO false alarms (2.70%), DALL-E 3 false negatives (48.03%), bimodal score distribution, and spatial/frequency complementarity (+82 unique correct predictions from FFT branch).
7. **Phase 7 (Standalone Inference Pipeline & CLI):** Implemented [`src/verixa/inference.py`](file:///c:/Verixa/src/verixa/inference.py) (`VerixaPredictor`) and [`scripts/predict.py`](file:///c:/Verixa/scripts/predict.py). Supports single-image, directory batch, manifest CSV, and submission predictions with sub-25 ms latency on RTX 4060. Verified with 6 unit tests in [`tests/test_inference.py`](file:///c:/Verixa/tests/test_inference.py).
8. **Phase 8 (Documentation, Polish & Final Verification):** Updated [`README.md`](file:///c:/Verixa/README.md), [`Phases.md`](file:///c:/Verixa/Phases.md), and [`Memory.md`](file:///c:/Verixa/Memory.md). All 68 automated unit tests pass cleanly (`pytest -v`), linter is 100% clean (`ruff check .`), working tree is clean.

---

## 3. Checkpoint & Model Registry

| Role | Architecture | Checkpoint Path | Best Epoch | Clean Acc | Clean AUROC | Clean FPR | Mean Transformed Acc | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Primary Champion** | Hybrid RGB+FFT (ConvNeXt-Tiny + 2D FFT CNN) | `models/convnext_tiny_hybrid_fft.pt` | Epoch 6 | **97.55%** | **99.68%** | **2.50%** | **96.95%** | **LOCKED & ACTIVE** |
| **Locked Fallback** | Robust RGB (ConvNeXt-Tiny, $p=0.8$ aug) | `models/convnext_tiny_robust_rgb.pt` | Epoch 3 | 96.83% | 99.64% | 2.53% | 96.11% | **LOCKED & PRESERVED** |
| *Baseline* | Clean RGB (ConvNeXt-Tiny, no aug) | `models/convnext_tiny_baseline.pt` | Epoch 2 | 97.03% | 99.64% | 2.80% | 88.34% | Historical Reference |
| *Ablation V2* | Hybrid V2 (Unfreeze Stage 2 + LS + Flip) | `models/convnext_tiny_hybrid_v2.pt` | Epoch 2 | 97.57% | 99.67% | 1.83% | 96.51% | Rejected Negative Result |
| *Ablation V3* | Hybrid V3 (Frozen S0..2 + LS + Flip) | `models/convnext_tiny_hybrid_v3.pt` | Epoch 2 | 96.80% | 99.65% | 3.77% | 96.41% | Rejected Negative Result |
| *Ablation V4* | Hybrid V4 (Frozen S0..2 + pos_weight=1.35) | `models/convnext_tiny_hybrid_v4_weighted.pt` | Epoch 3 | 96.53% | 99.63% | 2.07% | 96.03% | Rejected Negative Result |

---

## 4. Key Authoritative Metrics for Hybrid V4

To prevent any metric regression in documentation:
- **Clean Accuracy:** **96.53%**
- **Clean AUROC:** **99.63%**
- **Clean FPR:** **2.07%**
- **Mean Transformed Accuracy (16 conditions):** **96.03%**
- **Mean Transformed AUROC (16 conditions):** **99.45%**
- **Worst-Case Floor:** **94.75%** (`noise_sigma0.10`)
- **Composite Severe Accuracy:** **96.07%**
- **Best Epoch:** Epoch 3 (Early stopped at Epoch 7, runtime 685.9s).
- **Outcome:** FAILED promotion gate 1 (mean trans acc $\ge 96.85\%$) and gate 3 (worst-case floor $\ge 95.20\%$). Rejection confirmed.

---

## 5. Verification & Codebase Integrity

- **Automated Tests:** **68/68 passed in 8.40s** (`pytest -v`).
- **Linter:** **All checks passed** (`ruff check .`).
- **Clean Repository:** No benchmark data, raw training photos, or generated CSV dumps committed to Git.
- **Git History:** Clean, atomic phase commits with clear semantic messages.

---

## 6. Required Final Submission Files

When packaging for competition submission:
1. `models/convnext_tiny_hybrid_fft.pt` — Primary Champion Checkpoint (243.8 MB).
2. `models/convnext_tiny_robust_rgb.pt` — Locked Fallback Checkpoint (237.5 MB).
3. `scripts/predict.py` — Production standalone CLI interface.
4. `scripts/generate_submission.py` — Batch submission prediction generator.
5. `src/verixa/` — Complete core package (`models/`, `training/`, `evaluation/`, `inference.py`).
6. `reports/error_analysis.md` — Comprehensive Phase 6 forensic report.
7. `reports/threshold_calibration.json` — Canonical threshold calibration data.
8. `reports/final_hybrid_robustness_report.json` — Full 17-condition development metrics.
9. `README.md` — Complete master documentation with reproduction instructions.
10. `requirements.txt` & `pyproject.toml` — Dependency and build specifications.
