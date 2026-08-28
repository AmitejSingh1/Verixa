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
Train and evaluate a standard pretrained ConvNeXt-Tiny classifier on clean RGB images without transformation-aware augmentations to establish the unaugmented performance and vulnerability baseline.

### 2.2 Tasks
- [ ] Implement `src/verixa/models/convnext.py` wrapping torchvision's `convnext_tiny` with custom binary head.
- [ ] Implement PyTorch `Dataset` and `DataLoader` in `src/verixa/training/dataset.py` reading from `merged_manifest.csv`.
- [ ] Implement training loop in `src/verixa/training/trainer.py` supporting mixed precision (`torch.amp`), gradient accumulation, VRAM monitoring, and cosine learning rate scheduling.
- [ ] Implement clean baseline evaluation module in `src/verixa/evaluation/metrics.py` (Accuracy, AUROC, FPR).
- [ ] Create CLI training script `scripts/train_rgb.py` with configurable epochs, batch size, learning rate, and freeze levels.
- [ ] Train baseline ConvNeXt-Tiny for 5–10 epochs (clean data only, standard ImageNet normalization).
- [ ] Evaluate baseline on clean validation set and generate `reports/baseline_clean_eval.json`.
- [ ] Save checkpoint to `models/convnext_tiny_baseline.pt`.

### 2.3 Expected Outputs
- Trained baseline checkpoint `models/convnext_tiny_baseline.pt`.
- Baseline report `reports/baseline_clean_eval.json` with clean Accuracy, AUROC, and FPR.

### 2.4 Tests & Verification
- Unit tests for model forward pass, tensor shape verification, and metric calculation functions.
- Verify peak VRAM during training remains $< 6.0\text{ GB}$.

### 2.5 Completion Criteria
Baseline training completes cleanly without errors; checkpoint and validation metrics are saved and documented.

### 2.6 Commit Message
```text
feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training and evaluation
```

---

## Phase 3: Transformation-Aware Training + Decision Checkpoint #1

### 3.1 Objective
Implement dynamic, on-the-fly CPU-side data augmentation in the training pipeline to train a robust RGB ConvNeXt-Tiny model capable of maintaining high classification accuracy under severe distortions.

### 3.2 Tasks
- [ ] Implement `src/verixa/training/augmentations.py` with randomized transformations (JPEG $Q \in [30, 90]$, Gaussian blur $\sigma \in [0.5, 2.0]$, bilinear resize $0.25\times\text{--}0.5\times$, Gaussian noise $\sigma \in [0.02, 0.10]$, color jitter $\pm 20\%$, center crop $80\%$).
- [ ] Integrate transformation-aware pipeline into training DataLoader with configurable probability ($p=0.8$).
- [ ] Train robust ConvNeXt-Tiny model on augmented data using identical seed `1337` and hyperparameters.
- [ ] Save checkpoint to `models/convnext_tiny_robust_rgb.pt`.
- [ ] Evaluate robust model across clean validation data and all 6 required distortion suites at all severities.
- [ ] Compare robust RGB model against Phase 2 baseline; quantify degradation reduction ($\Delta_{\text{degradation}}$).
- [ ] Conduct **Decision Checkpoint #1**.

### 3.3 Decision Checkpoint #1 Criteria
- **Question:** Does transformation-aware training significantly improve accuracy/AUROC under severe JPEG ($Q=30$) and blur ($\sigma=2.0$) compared to the Phase 2 baseline?
- **Success Requirement:** Severe transformation accuracy must improve by $\ge 15\%$ relative to baseline with minimal clean-accuracy drop ($< 3\%$).
- **Status:** If successful, `models/convnext_tiny_robust_rgb.pt` is designated as the official **Fallback Submission Model**.

### 3.4 Expected Outputs
- Robust checkpoint `models/convnext_tiny_robust_rgb.pt`.
- Comparative report `reports/robust_rgb_vs_baseline.json`.

### 3.5 Commit Message
```text
feat(training): complete Phase 3 transformation-aware training and Checkpoint 1
```

---

## Phase 4: FFT Experiment + Decision Checkpoint #2

### 4.1 Objective
Design, train, and empirically evaluate an experimental dual-branch architecture (RGB ConvNeXt-Tiny + 2D FFT Magnitude Spectrum branch) to determine whether explicit frequency-domain features provide meaningful robustness advantages over spatial RGB features alone.

### 4.2 Tasks
- [ ] Implement `src/verixa/models/hybrid_fft.py` defining 2D FFT extraction, frequency convolution blocks, and spatial-spectral feature fusion.
- [ ] Create training CLI `scripts/train_fft.py` with gradient checkpointing support if needed for VRAM management.
- [ ] Train hybrid RGB+FFT model on the transformation-augmented dataset.
- [ ] Save checkpoint to `models/hybrid_fft_model.pt`.
- [ ] Evaluate hybrid model on clean validation set and under JPEG $Q \in \{90, 70, 50, 30\}$ and Gaussian Blur $\sigma \in \{0.5, 1.0, 2.0\}$.
- [ ] Conduct **Decision Checkpoint #2**.

### 4.3 Decision Checkpoint #2 Criteria
- **Decision Rule:**
  - **Adopt Hybrid Model:** IF the hybrid model demonstrates a statistically meaningful improvement ($\ge +1.5\%\text{ AUROC}$ or $\ge +2.0\%\text{ accuracy}$) on severe JPEG/blur distortions compared to the Phase 3 robust RGB model, AND fits within VRAM/latency constraints.
  - **Retain Robust RGB Model:** IF the hybrid model yields marginal, negligible, or negative improvements, or excessive VRAM/latency overhead. Document negative results transparently.
- **Outcome:** Fix the official primary architecture for remaining phases (`Final Model`).

### 4.4 Expected Outputs
- Hybrid model checkpoint `models/hybrid_fft_model.pt`.
- Experimental comparison report `reports/fft_experiment_decision.json`.

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

