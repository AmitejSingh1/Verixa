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
  - **Model Complexity Limit:** Maximum **< 2 Billion parameters** (our ConvNeXt-Tiny backbone is ~28.6M parameters, well within budget).
  - **Storage Ceiling:** Processed image storage target $< 500\text{ MB}$.

---

## 2. Current Development Status

- **Current Phase:** **Phase 3: Transformation-Aware Training + Decision Checkpoint #1 — COMPLETED & APPROVED**
- **Completed Work:**
  - **Phase 1 (Dataset Pipeline):** Fully complete and committed (`30c8775`).
  - **Phase 2 Pilot Experiment (8K Images):** Completed, validated VRAM usage (< 550 MB), and permanently preserved as historical artifacts.
  - **Final 30K Dataset Preparation:** Ingestion of 30,000 images, deduplication, deterministic 80/20 train/val split, and cross-split leakage verification are **COMPLETE AND FROZEN**.
  - **Official 30K Phase 2 Baseline Training:** **COMPLETED AND VERIFIED**. Trained clean ConvNeXt-Tiny on the frozen 30K dataset (`data/manifests/merged_manifest.csv`). Best Epoch 2 achieved **AUROC 0.9964**, **Accuracy 97.03%**, and **FPR 2.80%**. Checkpoint saved to `models/convnext_tiny_baseline.pt` and report saved to `reports/baseline_clean_eval.json`.
  - **Phase 3 Robust RGB Training & Evaluation:** **COMPLETED AND VERIFIED**. Trained robust ConvNeXt-Tiny with on-the-fly transformation augmentations ($p=0.8$) on the identical frozen 30K dataset (`data/manifests/merged_manifest.csv`, seed `1337`). Best Epoch 3 achieved **AUROC 0.9964**, **Clean Accuracy 96.83%**, and **FPR 2.53%**. Checkpoint saved to `models/convnext_tiny_robust_rgb.pt`.
  - **Decision Checkpoint #1:** **APPROVED**. Evaluated both models across all 16 benchmark conditions (clean + 15 distortion suites). Designated `models/convnext_tiny_robust_rgb.pt` as the official **Fallback Submission Model**.
- **In Progress / Current Stage:**
  - Git commit of Phase 3 deliverables and planning for Phase 4 (FFT Frequency Analysis).
- **Not Started:**
  - **Phase 4 (FFT Dual-Branch Experiment):** Prepared, awaiting user approval to begin implementation.
  - **Phase 5 (Held-Out Benchmark Evaluation):** Not started.

---

## 3. Dataset State (Current Frozen 30K Dataset)

The training and validation dataset is **FROZEN** and stored in [`data/manifests/merged_manifest.csv`](file:///c:/Verixa/data/manifests/merged_manifest.csv).

### 3.1 Composition & Breakdown
- **CIFAKE:** **15,000 images** (7,500 Authentic `0` / 7,500 Synthetic `1`) sourced from local KaggleHub cache (`data/processed/cifake/`).
- **SID_Set:** **15,000 images** (7,500 Authentic `0` / 7,500 Full Synthetic `1`) streamed via Hugging Face `saberzl/SID_Set` (`data/processed/sid_set/`).
- **Total Dataset Size:** Exactly **30,000 images**.
- **Class Balance:** Perfectly balanced **50.0% REAL (15,000) / 50.0% AI-GENERATED (15,000)**.
- **Exclusion of SID_Set Tampered Images:** During streaming, **7,250 tampered images (`label=2`)** were detected and strictly excluded.
- **Exclusion of WildFake:** WildFake (`hy2628982280/WildFake` on ModelScope) is **strictly excluded from the training dataset**. Technical rationale: ModelScope stores WildFake exclusively inside monolithic 16 GB to 51 GB `.zip` archives per architecture (`GAN_based.zip` 45.1 GB, `DALLE.zip` 24.4 GB, `Midjourney.zip` 51.1 GB). Individual HTTP retrieval returns `404`, making controlled sampling impossible without downloading tens of gigabytes and exceeding local disk budgets.

### 3.2 Standardization, Storage & Deterministic Split
- **Image Standardization:** All images are 3-channel RGB, bicubic resized to $224 \times 224$, compressed as JPEG ($Q=90$).
- **Actual On-Disk Storage:** **307.89 MB** (CIFAKE: 89.90 MB, SID_Set: 217.99 MB), well within the 500 MB budget.
- **Deterministic Split Methodology:** Stratified 80/20 split assigned using global seed `1337` and lexicographic SHA-256 group sorting:
  - **Train Split (`train`):** **23,999 images** (80.0%)
    - `CIFAKE Real (0)`: 6,000 | `CIFAKE Fake (1)`: 5,999
    - `SID_Set Real (0)`: 6,000 | `SID_Set Synthetic (1)`: 6,000
  - **Validation Split (`val`):** **6,001 images** (20.0%)
    - `CIFAKE Real (0)`: 1,500 | `CIFAKE Fake (1)`: 1,501
    - `SID_Set Real (0)`: 1,500 | `SID_Set Synthetic (1)`: 1,500
  - *(Note: The 1-image delta between 5,999 and 1,501 in CIFAKE Fake is due to duplicate hash grouping preserving exact identical files within the same split to avoid data leakage).*
- **Cross-Split Hash Leakage:** **Strictly 0**. ($\text{Train SHA} \cap \text{Val SHA} = \emptyset$).
- **Deduplication Findings:** 22 exact duplicate groups detected; 500 near-duplicate pairs reported in [`reports/final_30k_dedupe_report.json`](file:///c:/Verixa/reports/final_30k_dedupe_report.json).
- **Manifest Location:** [`data/manifests/merged_manifest.csv`](file:///c:/Verixa/data/manifests/merged_manifest.csv) (30,000 rows).
- **Dataset Statistics Report:** [`reports/final_dataset_stats.json`](file:///c:/Verixa/reports/final_dataset_stats.json).

---

## 4. 8K Pilot History & Preserved Artifacts

An earlier 8,000-image dataset (4,000 CIFAKE + 4,000 SID_Set; 6,401 train / 1,599 val) was ingested and used as a pilot to validate pipeline stability, mixed precision, and metrics.

### Preserved Historical Pilot Artifacts (NEVER OVERWRITE):
- **Model Checkpoint:** [`models/convnext_tiny_baseline_8k_pilot.pt`](file:///c:/Verixa/models/convnext_tiny_baseline_8k_pilot.pt) (226.5 MB)
- **Evaluation Report:** [`reports/baseline_clean_eval_8k_pilot.json`](file:///c:/Verixa/reports/baseline_clean_eval_8k_pilot.json)

### Historical Pilot Results (NOT the official baseline):
- **Best Epoch:** Epoch 3 (Early stopping triggered at Epoch 7 after 4 non-improving epochs)
- **Validation AUROC:** 0.9935 (99.35%)
- **Validation Accuracy:** 96.50%
- **Validation Loss:** 0.1259
- **Validation FPR:** 4.62% (37 false positives out of 800 authentic images)
- **FPR at 95% TPR:** 3.00%
- **Peak VRAM:** 545.83 MB allocated / 738.00 MB reserved
- **Average Epoch Duration:** ~36.8 seconds on RTX 4060 GPU

---

## 5. Model Benchmarks & Checkpoint Registry

### 5.1 Official Phase 2 Baseline (Clean Images Only)
- **Checkpoint:** [`models/convnext_tiny_baseline.pt`](file:///c:/Verixa/models/convnext_tiny_baseline.pt) (237.5 MB)
- **Report:** [`reports/baseline_clean_eval.json`](file:///c:/Verixa/reports/baseline_clean_eval.json) & [`reports/baseline_distortion_eval.json`](file:///c:/Verixa/reports/baseline_distortion_eval.json)
- **Best Epoch:** **Epoch 2** (Early stopped at Epoch 6)
- **Validation Metrics (Clean):**
  - **AUROC:** **0.9964 (99.64%)**
  - **Clean Accuracy:** **97.03%**
  - **Validation Loss:** **0.0920**
  - **FPR:** **2.80%** (84 FP / 3,000 Real)
  - **FNR:** **3.13%** (94 FN / 3,001 Fake)
  - **FPR at 95% TPR:** **1.63%**

### 5.2 Official Phase 3 Robust Model (Fallback Submission Model)
- **Checkpoint:** [`models/convnext_tiny_robust_rgb.pt`](file:///c:/Verixa/models/convnext_tiny_robust_rgb.pt) (237.5 MB)
- **Report:** [`reports/robust_rgb_clean_eval.json`](file:///c:/Verixa/reports/robust_rgb_clean_eval.json), [`reports/robust_distortion_eval.json`](file:///c:/Verixa/reports/robust_distortion_eval.json), and [`reports/robust_rgb_vs_baseline.json`](file:///c:/Verixa/reports/robust_rgb_vs_baseline.json)
- **Best Epoch:** **Epoch 3** (Early stopped at Epoch 7)
- **Training Setup:** Identical 30K dataset, random seed `1337`, $p=0.8$ CPU-side transformation augmentations (JPEG, blur, resize, noise, jitter, crop).
- **Validation Metrics (Clean):**
  - **AUROC:** **0.9964 (99.64%)** (identical to baseline)
  - **Clean Accuracy:** **96.83%** (only $-0.20$ percentage points penalty from clean baseline)
  - **Validation Loss:** **0.0991**
  - **FPR:** **2.53%** (76 FP / 3,000 Real — lower false alarm rate than baseline)
  - **FNR:** **3.80%** (114 FN / 3,001 Fake)
  - **FPR at 95% TPR:** **1.57%**

### 5.3 Decision Checkpoint #1 Comparative Findings
- **Transformation-aware training substantially improved robustness across every tested distortion family.**
- **Clean accuracy penalty was only 0.20 percentage points** ($97.03\% \rightarrow 96.83\%$), far below the $< 3.0\%$ degradation ceiling.
- **Severe Gaussian noise ($\sigma=0.10$) improved from 60.71% $\rightarrow$ 94.53% accuracy (+33.82 points)**, preventing catastrophic model failure.
- **Moderate Gaussian noise ($\sigma=0.05$) improved from 69.84% $\rightarrow$ 95.87% accuracy (+26.03 points)**.
- **Severe bilinear resize ($0.25\times$) improved from 88.19% $\rightarrow$ 94.33% accuracy (+6.14 points)**.
- **Severe resize False Positive Rate fell from 18.10% $\rightarrow$ 3.93%** (a $78\%$ reduction in false positive alarms).
- **Severe Gaussian blur ($\sigma=2.0$) False Positive Rate fell from 10.00% $\rightarrow$ 3.80%** (a $62\%$ reduction in false alarms).
- **Severe JPEG ($Q=30$)** accuracy improved from $92.90\% \rightarrow 96.00\%$ ($+3.10$ points) and loss halved ($0.2726 \rightarrow 0.1362$).
- **Worst-Case Transformed Floor:** The robust model's worst accuracy across all 15 distortion conditions was **94.33%**, versus **60.71%** for the unaugmented baseline.
- **Scientific Criterion Evaluation:** The predefined automated criterion requiring $\ge 15\%$ gain specifically on JPEG $Q=30$ and blur $\sigma=2.0$ was not literally met because the baseline was already high ($> 92\%$), making a $+15\%$ absolute gain mathematically impossible ($92.9\% + 15\% > 100\%$). However, the broader empirical evaluation conclusively demonstrated dramatic resilience across every condition where the baseline actually degraded.
- **Official Designation:** [`models/convnext_tiny_robust_rgb.pt`](file:///c:/Verixa/models/convnext_tiny_robust_rgb.pt) is officially accepted as the **Fallback Submission Model**.

---

## 6. Existing Code & Repository Structure

| Path | Purpose |
| :--- | :--- |
| [`PRD.md`](file:///c:/Verixa/PRD.md) | Product requirements, 30K dataset structure, transformation specs, success criteria |
| [`Architecture.md`](file:///c:/Verixa/Architecture.md) | System dataflow, storage budgets, model design, hardware constraints |
| [`Rules.md`](file:///c:/Verixa/Rules.md) | Core development rules, scientific rigor, command execution policies |
| [`Phases.md`](file:///c:/Verixa/Phases.md) | Sequential 8-phase milestones and verifiable completion checklists |
| [`src/verixa/models/convnext.py`](file:///c:/Verixa/src/verixa/models/convnext.py) | ConvNeXt-Tiny binary classifier implementation and backbone stage freezing logic |
| [`src/verixa/training/dataset.py`](file:///c:/Verixa/src/verixa/training/dataset.py) | PyTorch Dataset, custom transform support, and eval DataLoader factories |
| [`src/verixa/training/augmentations.py`](file:///c:/Verixa/src/verixa/training/augmentations.py) | CPU-side 6-family augmentation engine and pickle-safe discrete distortion suites |
| [`src/verixa/training/trainer.py`](file:///c:/Verixa/src/verixa/training/trainer.py) | AMP training loop, in-place ASCII progress display, VRAM tracker, early stopping |
| [`src/verixa/evaluation/metrics.py`](file:///c:/Verixa/src/verixa/evaluation/metrics.py) | Evaluation engine: Accuracy, AUROC, FPR at threshold, FPR at 95% TPR, confusion matrix |
| [`src/verixa/data/ingestion.py`](file:///c:/Verixa/src/verixa/data/ingestion.py) | Local image tree ingestion with bicubic resizing, hashing, and manifest creation |
| [`src/verixa/data/hf_ingestion.py`](file:///c:/Verixa/src/verixa/data/hf_ingestion.py) | Hugging Face streaming ingestion with buffer shuffling and label mapping |
| [`src/verixa/data/manifest.py`](file:///c:/Verixa/src/verixa/data/manifest.py) | Manifest parsing, merging, deterministic 80/20 splitting, and statistics reporting |
| [`src/verixa/data/dedupe.py`](file:///c:/Verixa/src/verixa/data/dedupe.py) | SHA-256 exact deduplication and dHash perceptual hash duplicate analyzer |
| [`scripts/train_rgb.py`](file:///c:/Verixa/scripts/train_rgb.py) | CLI training script for clean baseline and robust RGB models |
| [`scripts/evaluate_robustness.py`](file:///c:/Verixa/scripts/evaluate_robustness.py) | CLI 16-condition distortion evaluation harness with comparative delta reporting |
| [`scripts/ingest_local_dataset.py`](file:///c:/Verixa/scripts/ingest_local_dataset.py) | CLI script for ingesting local datasets (e.g. CIFAKE) |
| [`scripts/ingest_hf_dataset.py`](file:///c:/Verixa/scripts/ingest_hf_dataset.py) | CLI script for streaming Hugging Face datasets (e.g. SID_Set) |
| [`tests/test_augmentations.py`](file:///c:/Verixa/tests/test_augmentations.py) | Unit tests for all 6 augmentation families and evaluation suites |
| [`tests/test_convnext.py`](file:///c:/Verixa/tests/test_convnext.py) | Unit tests for ConvNeXt forward pass, freezing, and binary metrics |
| [`tests/test_manifest.py`](file:///c:/Verixa/tests/test_manifest.py) | Unit tests for deterministic splitting and manifest statistics |
| [`tests/test_dedupe.py`](file:///c:/Verixa/tests/test_dedupe.py) | Unit tests for hash deduplication and split leakage detection |
| [`tests/test_schema.py`](file:///c:/Verixa/tests/test_schema.py) | Unit tests for label parsing and validation |

---

## 7. Verification Status

- **Automated Tests:** **46/46 passed** (`pytest -v`).
- **Linter & Formatting:** **All checks passed** (`ruff check .`).
- **Compute Environment:** Python 3.11.16, PyTorch 2.6.0+cu128, CUDA 12.8 active on Windows.
- **Pretrained Weights Cached:** `convnext_tiny-983f1562.pth` exists locally in Torch hub cache (`114.4 MB`).
- **GPU Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,192 MB VRAM, measured peak ~546 MB).

---

## 8. Critical Decisions Already Made (Do NOT Silently Reverse)

1. **ConvNeXt-Tiny is the Primary Backbone:** Selected for parameter efficiency (~28.6M params) and high feature representational capacity.
2. **Fast Fourier Transform (FFT) is Experimental:** Belongs strictly to Phase 4. It will only be adopted if it demonstrates a statistically significant robustness delta over the spatial model.
3. **No Training From Scratch:** We use ImageNet-1K pretrained weights and fine-tune Stage 3 and the classification head.
4. **No Ensembles or Giant Models:** Stay strictly under the 2B parameter constraint and 8 GB VRAM budget.
5. **Held-Out Benchmark Isolation:** COCO val2017 (authentic) and DALL-E Advanced (synthetic) in `data/held_out_benchmark/` must **never** be used for training, validation tuning, threshold calibration, or early stopping. They are reserved strictly for Phase 5.
6. **No WildFake in Training:** Excluded due to ModelScope's monolithic 16–51 GB zip archive structure.
7. **Strict Head-to-Head Coupling:** Phase 3 and Phase 4 experiments must use the **exact same frozen 30K dataset and split** as the Phase 2 baseline so that the measured robustness delta is scientifically valid.
8. **Preserve 8K Pilot Artifacts:** `models/convnext_tiny_baseline_8k_pilot.pt` and `reports/baseline_clean_eval_8k_pilot.json` must never be deleted or overwritten.
9. **Preserve Fallback Submission Model:** `models/convnext_tiny_robust_rgb.pt` is locked as our official fallback submission model and must never be overwritten or modified.
10. **Never Fabricate Metadata:** Generator architecture tags must only be recorded if present in verified dataset features.

---

## 9. AI-Agent Workflow & User Collaboration Rules

- **User Runs Training Commands:** The user wants to personally execute long-running training commands locally in their VS Code PowerShell terminal to view live progress.
- **Never Silently Train in Background:** Do not launch multi-epoch training commands via background agent tasks without explicit user instruction.
- **Provide Exact Commands:** When prompting the user, provide the complete, copy-pasteable PowerShell command using the direct Python binary (`& 'C:\Users\amite\anaconda3\envs\verixa\python.exe' ...`) to avoid `conda run` buffering issues.
- **ASCII-Only Progress Display:** Windows PowerShell mangles Unicode block characters (`███` $\rightarrow$ `â–ˆ`). Always use 7-bit ASCII characters (`#` and `-`).
- **Single-Line In-Place Rendering:** Terminal output must use `\r` to update progress on a single line, followed by a permanent epoch summary line upon stage completion. Do not print a new line for every batch.

---

## 10. Git State

- **Completed Commits:**
  - `30c8775` — `feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training and evaluation`
  - `c5062b9` — `feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training on 30K dataset`
- **Current Working Tree:** All Phase 3 deliverables complete and verified. Ready to commit:
  `feat(training): complete Phase 3 transformation-aware training and Checkpoint 1`

---

## 11. NEXT ACTION

1. Stage all Phase 3 implementation, tests, reports, and documentation.
2. Commit with message: `feat(training): complete Phase 3 transformation-aware training and Checkpoint 1`.
3. Verify `git status` is 100% clean.
4. Prepare Phase 4: FFT-only branch implementation, empirical evaluation on clean + JPEG + blur, and rigorous hypothesis testing before building any hybrid fusion model.

---

## 12. Incoming Agent Checklist

When you resume or take over this repository:
- [ ] Read `Memory.md` first.
- [ ] Read `Rules.md` for strict behavioral guidelines.
- [ ] Read `Phases.md` for milestone requirements.
- [ ] Run `git status` to verify the exact state of the working tree.
- [ ] Run `pytest -v` and `ruff check .` before modifying any code.
- [ ] Never assume an intended experiment has finished without checking files on disk.
- [ ] Follow the exact Next Action outlined in Section 11.
