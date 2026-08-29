# Development Phases & Milestones — Verixa

This document defines the strict eight-phase development roadmap for Verixa. Each phase is sequential, verifiable, and atomic.

---

## Phase 1: Repo Scaffolding & Dataset Pipeline

### 1.1 Objective
Construct a robust, modular, and leak-free dataset ingestion and normalization pipeline that standardizes multi-source image datasets (SID_Set, CIFAKE, WildFake) into 224×224 compressed JPEGs with complete provenance manifests and deterministic train/validation splits.

### 1.2 Tasks
- [x] Verify Conda environment (`verixa`), PyTorch CUDA 12.8, and RTX 4060 GPU availability.
- [x] Inspect source dataset schemas and confirm label mappings for SID_Set and CIFAKE.
- [x] Create `config/cifake_label_map.json` and verify `config/sid_set_label_map.binary.json`.
- [x] Build and test `src/verixa/data/wildfake_ingestion.py` using ModelScope streaming metadata.
- [x] Ingest CIFAKE pilot (3,000 images: 1,500 Real / 1,500 Fake) $\rightarrow$ `data/processed/cifake/`.
- [x] Ingest SID_Set pilot (3,000 images: 1,500 Real / 1,500 Synthetic) $\rightarrow$ `data/processed/sid_set/`.
- [ ] Ingest WildFake pilot (2,100 images: 1,000 Real / 1,100 AI across 5 architectures) $\rightarrow$ `data/processed/wildfake/`.
- [ ] Merge individual CSV manifests into unified `data/manifests/merged_manifest.csv`.
- [ ] Execute exact duplicate detection and perceptual hash leakage checks.
- [ ] Assign deterministic `train` (80%) and `val` (20%) splits using fixed seed `1337` and SHA-256 lexicographic sorting.
- [ ] Save reproducible configuration to `config/pilot_sampling_config.json`.
- [ ] Generate comprehensive dataset statistics report (`reports/pilot_dataset_stats.json`).

### 1.3 Expected Outputs
- `data/processed/` containing ~8,100 normalized 224×224 JPEGs (< 100 MB total disk usage).
- `data/manifests/merged_manifest.csv` containing 8,100 rows with complete metadata.
- `reports/pilot_dataset_stats.json` with verified class balance and split distributions.

### 1.4 Tests & Verification
- `pytest -v tests/test_schema.py tests/test_manifest.py tests/test_dedupe.py tests/test_wildfake_ingestion.py`
- Verify zero split leakage (no duplicate SHA-256 digests across train and val splits).
- `ruff check .` clean.

### 1.5 Completion Criteria
All 8,100 pilot images ingested, verified non-corrupt, unified manifest validated, zero leakage confirmed, and full test suite passing.

### 1.6 Commit Message
```text
feat(data): complete Phase 1 dataset ingestion, manifests, and deterministic split
```

---

## Phase 2: RGB ConvNeXt-Tiny Baseline

### 2.1 Objective
Train and evaluate a standard pretrained ConvNeXt-Tiny classifier on clean RGB images without transformation-aware augmentations or frequency features, establishing the unaugmented performance baseline.

The Phase 2 workflow consists of two structured steps:
1. **8K Pilot Experiment (Completed & Preserved):** Validated pipeline, AMP stability, peak VRAM (< 550 MB), and metrics. Checkpoint permanently preserved as `models/convnext_tiny_baseline_8k_pilot.pt` and report preserved as `reports/baseline_clean_eval_8k_pilot.json`.
2. **Final Frozen 30K Dataset & Official Baseline:** The dataset is scaled to 30,000 images (15,000 CIFAKE + 15,000 SID_Set; 24,000 train / 6,000 val, seed `1337`) and frozen. The clean ConvNeXt-Tiny baseline is trained on this exact final dataset to establish the official unaugmented baseline (`models/convnext_tiny_baseline.pt`, `reports/baseline_clean_eval.json`) for direct comparison with Phase 3.

### 2.2 Tasks
- [x] Implement `src/verixa/models/convnext.py` wrapping torchvision's `convnext_tiny(weights=IMAGENET1K_V1)` with custom binary classification head.
- [x] Implement PyTorch `Dataset` and `DataLoader` in `src/verixa/training/dataset.py` reading from `merged_manifest.csv` with standard ImageNet normalization.
- [x] Implement training engine in `src/verixa/training/trainer.py` supporting mixed precision (`torch.amp`), peak VRAM monitoring, cosine annealing scheduler, best-model selection, and early stopping.
- [x] Implement clean baseline evaluation module in `src/verixa/evaluation/metrics.py` (Accuracy, AUROC, FPR at threshold and 95% TPR).
- [x] Create CLI training script `scripts/train_rgb.py` supporting configurable epochs, batch size, learning rate, freeze levels, and early stopping patience.
- [x] **8K Pilot Validation:** Ran smoke test and pilot baseline training; verified VRAM (< 550 MB) and metrics; preserved outputs to `models/convnext_tiny_baseline_8k_pilot.pt` and `reports/baseline_clean_eval_8k_pilot.json`.
- [x] **Final 30K Dataset Ingestion & Freeze:** Ingest 15,000 CIFAKE + 15,000 SID_Set (50/50 balance); run deduplication and zero-leakage check; assign deterministic 80/20 split (24,000 train / 6,000 val, seed `1337`); freeze `data/manifests/merged_manifest.csv`.
- [x] **Official 30K Baseline Training:** Train clean ConvNeXt-Tiny on the final 30K training split for a maximum of 20 epochs with early stopping (patience = 4 epochs without validation AUROC improvement).
- [x] Model selection based strictly on highest validation AUROC on the 6,000 validation split. The held-out COCO/DALL-E benchmark must NEVER be used for early stopping, tuning, or model selection.
- [x] Save the official best validation-AUROC checkpoint to `models/convnext_tiny_baseline.pt`.
- [x] Evaluate the best checkpoint on the clean 6,000 validation set and record the best epoch and corresponding metrics in `reports/baseline_clean_eval.json`.

### 2.3 Official Training Configuration
- **Dataset:** 30K CIFAKE + SID_Set (`data/manifests/merged_manifest.csv`)
- **Split:** Fixed deterministic split (24,000 train / 6,000 val, seed `1337`)
- **Backbone:** ConvNeXt-Tiny (`IMAGENET1K_V1`), Stages 0–2 frozen initially, Stage 3 + binary head trained
- **Max Epochs:** 20
- **Early Stopping:** Patience = 4 epochs without validation AUROC improvement
- **Model Selection Metric:** Validation AUROC
- **Batch Size:** 32
- **Optimizer:** AdamW ($\text{lr}=10^{-4}$, weight decay $10^{-2}$)
- **Scheduler:** CosineAnnealingLR ($\eta_{\min}=10^{-6}$)
- **Precision:** Mixed precision (`torch.amp` FP16)
- **Augmentations:** None (clean ImageNet normalization only, no distortion transforms, no FFT)

### 2.4 Expected Outputs
- **Historical Pilot Deliverables (Preserved):**
  - Checkpoint: `models/convnext_tiny_baseline_8k_pilot.pt`
  - Diagnostic report: `reports/baseline_clean_eval_8k_pilot.json`
- **Official Phase 2 Deliverables (Final Baseline):**
  - Best model checkpoint: `models/convnext_tiny_baseline.pt`
  - Official evaluation report: `reports/baseline_clean_eval.json` (recording best epoch, Accuracy, AUROC, FPR, and peak VRAM)

### 2.5 Tests & Verification
- Unit tests for model forward pass, tensor shapes, stage freezing, and metric functions (`tests/test_convnext.py`).
- Verify peak VRAM during training remains $< 6.0\text{ GB}$ (budget hard ceiling: 8.0 GB).
- Verify zero use of held-out COCO/DALL-E benchmark during Phase 2.

### 2.6 Completion Criteria
1. 30K dataset ingested, deduplicated, split deterministically (24K train / 6K val, seed `1337`), and verified leak-free.
2. Official baseline training on 30K dataset completes (either reaching 20 epochs or early-stopping after 4 epochs of non-improving validation AUROC).
3. Best-epoch checkpoint is saved to `models/convnext_tiny_baseline.pt` embedding complete reproducibility configuration.
4. Official baseline report `reports/baseline_clean_eval.json` is generated documenting the best epoch's clean metrics.
5. All automated unit tests (`pytest -v`) and linter (`ruff check .`) pass cleanly.

### 2.7 Commit Message
```text
feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training and evaluation
```

---

## Phase 3: Transformation-Aware Training + Decision Checkpoint #1

### 3.1 Objective
Implement dynamic, on-the-fly CPU-side data augmentation in the training pipeline to train a robust RGB ConvNeXt-Tiny model capable of maintaining high classification accuracy under severe distortions. Training uses the exact same frozen 30K dataset and split (`data/manifests/merged_manifest.csv`: 24,000 train / 6,000 val, seed `1337`) to ensure strict head-to-head experimental validity against the Phase 2 baseline.

### 3.2 Tasks
- [x] Implement `src/verixa/training/augmentations.py` with randomized transformations (JPEG $Q \in [30, 90]$, Gaussian blur $\sigma \in [0.5, 2.0]$, bilinear resize $0.25\times\text{--}0.5\times$, Gaussian noise $\sigma \in [0.02, 0.10]$, color jitter $\pm 20\%$, center crop $80\%$).
- [x] Integrate transformation-aware pipeline into training DataLoader with configurable probability ($p=0.8$).
- [x] Train robust ConvNeXt-Tiny model on augmented data using the exact same 30K split, seed `1337`, and hyperparameters.
- [x] Save checkpoint to `models/convnext_tiny_robust_rgb.pt`.
- [x] Evaluate robust model across clean validation data and all 6 required distortion suites at all severities.
- [x] Compare robust RGB model against Phase 2 baseline; quantify degradation reduction ($\Delta_{\text{degradation}}$).
- [x] Conduct **Decision Checkpoint #1**.

### 3.3 Decision Checkpoint #1 Results & Status
- **Question:** Does transformation-aware training significantly improve robustness under severe distortions compared to the Phase 2 baseline without compromising clean accuracy?
- **Scientific Findings:**
  - **Clean Performance:** Clean accuracy penalty was only **$0.20$ percentage points** ($97.03\% \rightarrow 96.83\%$), well within the $< 3.0\%$ degradation ceiling. Clean AUROC remained identical at **$0.9964$ ($99.64\%$)**, and clean FPR improved from $2.80\%$ to **$2.53\%$**.
  - **Severe Noise ($\sigma=0.10$):** Improved from **$60.71\% \rightarrow 94.53\%$ accuracy ($+33.82$ points)**, preventing catastrophic classification collapse.
  - **Moderate Noise ($\sigma=0.05$):** Improved from **$69.84\% \rightarrow 95.87\%$ accuracy ($+26.03$ points)**.
  - **Severe Bilinear Resize ($0.25\times$):** Improved from **$88.19\% \rightarrow 94.33\%$ accuracy ($+6.14$ points)**; False Positive Rate plummeted from **$18.10\% \rightarrow 3.93\%$** (a $78\%$ reduction in false alarms).
  - **Severe Gaussian Blur ($\sigma=2.0$):** Accuracy improved from $92.67\% \rightarrow 95.23\%$ ($+2.56$ points); False Positive Rate fell from **$10.00\% \rightarrow 3.80\%$** (a $62\%$ reduction in false alarms).
  - **Severe JPEG ($Q=30$):** Accuracy improved from $92.90\% \rightarrow 96.00\%$ ($+3.10$ points); loss dropped by $50\%$ ($0.2726 \rightarrow 0.1362$).
  - **Worst-Case Floor:** The robust model's worst-case transformed accuracy across all 15 distortion conditions was **$94.33\%$**, compared to **$60.71\%$** for the unaugmented baseline.
  - **Checkpoint Criterion Note:** The predefined automated criterion requiring $\ge 15\%$ gain specifically on JPEG $Q=30$ and blur $\sigma=2.0$ was not literally met because the baseline was already high ($> 92\%$), making a $+15\%$ absolute gain mathematically impossible ($92.9\% + 15\% > 100\%$). However, the broader empirical evaluation conclusively demonstrates dramatic resilience across every condition where the baseline actually degraded.
- **Decision:** **APPROVED.** `models/convnext_tiny_robust_rgb.pt` is officially accepted and designated as the **Fallback Submission Model**.

### 3.4 Expected Outputs
- Robust checkpoint `models/convnext_tiny_robust_rgb.pt` (verified, Epoch 3 best, AUROC 0.9964).
- Clean validation report `reports/robust_rgb_clean_eval.json`.
- Comparative report `reports/robust_rgb_vs_baseline.json` & `reports/robust_distortion_eval.json`.

### 3.5 Commit Message
```text
feat(training): complete Phase 3 transformation-aware training and Checkpoint 1
```

---

## Phase 4: FFT Experiment + Decision Checkpoint #2

### 4.1 Objective
Design, train, and empirically evaluate an experimental dual-branch architecture (RGB ConvNeXt-Tiny + 2D FFT Magnitude Spectrum branch) to determine whether explicit frequency-domain features provide meaningful robustness advantages over spatial RGB features alone.

### 4.2 Tasks
- [x] Implement `src/verixa/models/fft.py` defining 2D FFT extraction (`torch.fft.fft2` -> `fftshift` -> log-magnitude standardized) and standalone `FFTClassifier` (~1.63M parameters).
- [x] Create training CLI `scripts/train_fft_standalone.py` and evaluate standalone FFT model on clean, JPEG, and Gaussian blur.
- [x] Implement `src/verixa/models/hybrid.py` (`HybridRGBFFTClassifier`) with direct concatenation fusion ($768 + 512 = 1,280\text{-d}$) and stage freezing.
- [x] Create training CLI `scripts/train_hybrid.py` and train on identical 30K dataset with $p=0.8$ augmentations (seed `1337`).
- [x] Save checkpoint to `models/convnext_tiny_hybrid_fft.pt`.
- [x] Evaluate hybrid model across full 16-condition benchmark suite (`reports/hybrid_distortion_eval.json`).
- [x] Conduct **Decision Checkpoint #2** comparing hybrid against locked fallback `models/convnext_tiny_robust_rgb.pt`.

### 4.3 Decision Checkpoint #2 Results & Status
- **Question:** Does the hybrid RGB+FFT model provide meaningful, consistent robustness and generalization advantages over the locked robust RGB fallback model?
- **Empirical Findings:**
  1. **Standalone FFT Viability:** Standalone FFT demonstrated genuine discriminative capacity from scratch (**84.24% accuracy**, **94.65% AUROC** on clean data), and preserved high AUROC under severe JPEG $Q=30$ (**93.49%**), though it degraded under severe blur $\sigma=2.0$ (**73.29% accuracy**, **29.57% FPR**).
  2. **Hybrid Superiority Across All 16 Conditions:** The simple direct concatenation hybrid outperformed the robust RGB fallback across **16 out of 16 evaluated conditions**:
     - **Clean Accuracy:** **97.55%** vs. 96.83% fallback (**+0.72%**).
     - **Clean AUROC:** **99.68%** vs. 99.64% fallback (**+0.04%**).
     - **Mean Transformed Accuracy:** **96.96%** vs. 96.11% fallback (**+0.85% average gain** across 15 distortions).
     - **Mean Transformed AUROC:** **99.56%** vs. 99.40% fallback (**+0.16%**).
     - **Worst-Case Accuracy Floor:** Elevated from 94.33% to **95.40%** (**+1.07% higher robustness floor**).
     - **Severe Distortions:** JPEG $Q=30$ improved to **96.77%** (+0.77%), Blur $\sigma=2.0$ improved to **96.68%** (+1.45%), Resize $0.25\times$ improved to **96.27%** (+1.94%), Noise $\sigma=0.10$ improved to **95.40%** (+0.87%).
     - **False Alarm Suppression:** FPR on severe blur fell from $3.80\% \rightarrow 2.83\%$; FPR on severe resize fell from $3.93\% \rightarrow 2.57\%$; clean FPR@95% TPR fell from $1.57\% \rightarrow 1.03\%$.
- **Decision:** **ACCEPTED.**
  - [`models/convnext_tiny_hybrid_fft.pt`](file:///c:/Verixa/models/convnext_tiny_hybrid_fft.pt) is designated as the **Current Best Candidate (Primary Submission Model)**.
  - [`models/convnext_tiny_robust_rgb.pt`](file:///c:/Verixa/models/convnext_tiny_robust_rgb.pt) remains permanently locked as the official **Fallback Submission Model**.

### 4.4 Expected Outputs
- Hybrid model checkpoint `models/convnext_tiny_hybrid_fft.pt` (Epoch 6 best, AUROC 0.9968).
- Standalone FFT checkpoint `models/fft_standalone.pt` (Epoch 8 best, AUROC 0.9465).
- Experimental decision report `reports/fft_experiment_decision.json`.
- Comprehensive 16-condition report `reports/hybrid_distortion_eval.json`.

### 4.5 Commit Message
```text
feat(exp): complete Phase 4 FFT dual-branch experiment and Checkpoint 2
```

---

## Phase 5: Final Robustness Evaluation & Held-Out Benchmark

### 5.1 Objective
Execute comprehensive, rigorous evaluation of the final selected model across all required real-world transformations and severities, followed by a single-pass evaluation on the isolated held-out benchmark.

### 5.2 Tasks
- [ ] Implement comprehensive evaluation harness `scripts/evaluate_robustness.py`.
- [ ] Run evaluation on clean validation set (Accuracy, AUROC, FPR).
- [ ] Run evaluation across all 14 individual transformation conditions:
  - JPEG ($Q=90, 70, 50, 30$)
  - Gaussian Blur ($\sigma=0.5, 1.0, 2.0$)
  - Resize ($0.5\times, 0.25\times$)
  - Gaussian Noise ($\sigma=0.02, 0.05, 0.10$)
  - Color Jitter ($\pm 20\%$)
  - Center Crop ($80\%$)
- [ ] Calculate robustness degradation curves for every distortion type.
- [ ] Ingest and evaluate isolated held-out benchmark (COCO val2017 authentic + DALL-E Advanced synthetic).
- [ ] Compile full results into `reports/final_robustness_benchmark.json` and Markdown summary.

### 5.3 Expected Outputs
- `reports/final_robustness_benchmark.json` containing complete evaluation tables and metrics.
- `reports/held_out_benchmark_results.json` containing final benchmark scores.

### 5.4 Commit Message
```text
feat(eval): complete Phase 5 comprehensive robustness and held-out benchmark evaluation
```

---

## Phase 6: Error Analysis & Diagnostic Breakdown

### 6.1 Objective
Perform in-depth failure mode analysis to discover specific patterns where the model misclassifies imagery, breaking down performance by generator architecture, distortion level, and image characteristics.

### 6.2 Tasks
- [ ] Implement error analysis diagnostic script `scripts/error_analysis.py`.
- [ ] Breakdown False Positive (FP) cases: analyze which authentic images trigger false alarms.
- [ ] Breakdown False Negative (FN) cases: analyze which generator architectures (DDPM, BigGAN, LDM, DALL-E, GLIDE) evade detection.
- [ ] Correlate error rates with distortion severity (e.g., error rate vs. JPEG quality level).
- [ ] Extract and save representative failure cases to `reports/failure_cases/`.
- [ ] Compile comprehensive diagnostic report `reports/error_analysis_report.md`.

### 6.3 Expected Outputs
- `reports/error_analysis_report.md` with detailed breakdown tables and actionable insights.

### 6.4 Commit Message
```text
feat(analysis): complete Phase 6 error analysis and failure mode breakdown
```

---

## Phase 7: Standalone Inference Pipeline & CLI

### 7.1 Objective
Package the final trained model into a self-contained, production-ready inference module and CLI that generates calibrated predictions and structured JSON outputs for single images or directories.

### 7.2 Tasks
- [ ] Implement `src/verixa/inference.py` defining `VerixaPredictor` class with batching and automatic device detection.
- [ ] Build standalone CLI `scripts/predict.py` supporting `--image`, `--dir`, `--output-json`, and `--threshold` flags.
- [ ] Verify inference latency ($< 50\text{ ms}$ per image on RTX 4060 GPU).
- [ ] Add unit and integration tests in `tests/test_inference.py`.

### 7.3 Expected Outputs
- Working CLI `scripts/predict.py`.
- Automated tests verifying deterministic JSON outputs across sample images.

### 7.4 Commit Message
```text
feat(inference): complete Phase 7 standalone inference pipeline and CLI
```

---

## Phase 8: Documentation, Polish & Submission Packaging

### 8.1 Objective
Polish all repository documentation, format benchmarks into GitHub markdown tables, verify total reproducibility from scratch, and prepare the final competition submission package.

### 8.2 Tasks
- [ ] Update `README.md` with complete architecture diagrams, final benchmark tables, quickstart guide, and citation information.
- [ ] Review and clean `requirements.txt` and `pyproject.toml`.
- [ ] Verify fresh environment installation and run end-to-end smoke test.
- [ ] Package final report summaries and presentation artifacts.

### 8.3 Expected Outputs
- Polished, comprehensive `README.md`.
- Verified, fully reproducible repository ready for judging and evaluation.

### 8.4 Commit Message
```text
docs(polish): complete Phase 8 documentation, report summaries, and submission polish
```

