# Verixa: Forensic AI-Generated Image Detection Under Real-World Transformations

[![CI / Quality Checks](https://img.shields.io/badge/pytest-68%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA-EE4C2C.svg)]()
[![Model](https://img.shields.io/badge/Architecture-Hybrid%20ConvNeXt--Tiny%20%2B%202D%20FFT-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Verixa is a production-grade, forensic computer vision system engineered to detect AI-generated imagery across diverse generative architectures (GANs, Latent Diffusion, DiT) while remaining resilient against aggressive real-world digital distortions (JPEG recompression, Gaussian blurring, severe downsampling, Gaussian noise, color jittering, and cropping).

---

## Architecture Overview

Verixa adopts a **Dual-Branch Hybrid Fusion Architecture** that combines spatial perceptual representations with explicit frequency-domain spectral analysis:

```
                                  +---------------------------------------+
                                  |         Input Image (224x224)         |
                                  +---------------------------------------+
                                       /                             \
                                      /                               \
                     [RGB Stream]    /                                 \    [Spectral Stream]
                                    v                                   v
             +-------------------------------+             +-------------------------------+
             |      ConvNeXt-Tiny Backbone   |             |     2D FFT Spectrum Extractor |
             |   (Stages 0..2 Frozen, S3 ON) |             |  (fft2 -> fftshift -> log1p)  |
             +-------------------------------+             +-------------------------------+
                            |                                              |
                     768-d Spatial Vector                          512-d Spectral Vector
                            \                                              /
                             \                                            /
                              +------------------------------------------+
                              | Concatenation Head (1,280-d Feature Map) |
                              +------------------------------------------+
                                                   |
                                             Linear (256-d)
                                                   |
                                             Dropout (p=0.3)
                                                   |
                                            Linear (1 logit)
                                                   |
                                           Sigmoid Activation
                                                   |
                                                   v
                                 Synthetic Probability in [0.0, 1.0]
                                (Class 0: Authentic | Class 1: Synthetic)
```

1. **Spatial Backbone (ConvNeXt-Tiny):** Pretrained on ImageNet-1K (`IMAGENET1K_V1`). Stages 0 through 2 remain strictly frozen to preserve universal semantic primitives, while Stage 3 is fine-tuned to capture generator-specific high-level visual incoherence (plasticized skin, anatomical anomalies, non-physical shadows).
2. **Spectral Backbone (2D FFT CNN Branch):** Computes the centered 2D fast Fourier transform magnitude spectrum ($|F(u, v)|$), normalized with $\log(1 + |F|)$ and standardized. A 4-stage convolutional neural network compresses the frequency spectrum into a 512-dimensional feature embedding, capturing high-frequency periodic grids, checkerboard artifacts from upsampling convolutions, and high-frequency spectral attenuation.
3. **Fusion Classifier Head:** Concatenates the 768-d spatial and 512-d spectral vectors into a 1,280-d joint representation, followed by a 256-d projection layer with Dropout ($p=0.3$) and single-logit classification.

---

## Key Experimental Results

All development evaluations are executed on a fixed, leak-free **6,001-image validation split** (23,999 train / 6,001 val, seed `1337`) drawn from 30,000 processed images (CIFAKE + SID_Set).

### 1. Development Performance & 17-Condition Robustness Suite

| Model Configuration | Status | Clean Acc | Clean AUROC | Clean FPR | Mean Transformed Acc (16 Conds) | Worst-Case Floor | Composite Severe Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2 Clean Baseline** | Ablation Baseline | 97.03% | 99.64% | 2.80% | 88.34% | 60.71% (`noise_0.10`) | 89.12% |
| **Phase 3 Robust RGB** | **Locked Fallback** | 96.83% | 99.64% | 2.53% | 96.11% | 94.33% (`resize_0.25`) | 96.33% |
| **Hybrid V1 (RGB + FFT)** | **Champion Model** | **97.55%** | **99.68%** | **2.50%** | **96.95%** | **95.37%** (`noise_0.10`) | **97.05%** |
| **Hybrid V2** (Unfreeze S2+LS+Flip) | Rejected Candidate | 97.57% | 99.67% | 1.83% | 96.51% | 93.75% (`noise_0.10`) | 96.52% |
| **Hybrid V3** (Clean Ablation: LS+Flip) | Rejected Candidate | 96.80% | 99.65% | 3.77% | 96.41% | 94.55% (`noise_0.10`) | 96.48% |
| **Hybrid V4** (`pos_weight=1.35`) | Rejected Candidate | 96.53% | 99.63% | 2.07% | 96.03% | 94.75% (`noise_0.10`) | 96.07% |

### 2. Held-Out Benchmark Performance ($N = 13,841$)

The held-out benchmark contains **4,998 authentic COCO val2017 photographs** and **8,843 modern synthetic DALL·E 3 images**. Evaluated once under strict zero-shot quarantine:

| Model | Checkpoint Path | Overall Acc | AUROC | COCO Accuracy (Real) | COCO FPR | DALL·E 3 Recall (Fake) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RGB+FFT (V1)** | `models/convnext_tiny_hybrid_fft.pt` | **68.34%** | **92.13%** | **97.30%** | **2.70%** | **51.97%** |
| **Robust RGB (Fallback)** | `models/convnext_tiny_robust_rgb.pt` | 66.79% | 90.74% | 94.36% | 5.64% | 51.20% |
| **Net Advantage ($\Delta$)** | — | **+1.55%** | **+1.39%** | **+2.94%** | **-2.94%** | **+0.77%** |

**Key Takeaway:** The Hybrid V1 champion model reduced false positive alarms on authentic photographs by **52.1%** ($5.64\% \rightarrow 2.70\%$) while simultaneously detecting $+68$ additional DALL·E 3 synthetic images and lifting AUROC above $92\%$.

---

## Why Specific Techniques Succeeded or Failed

### 1. Why Transformation-Aware Augmentation Succeeded (Phase 3)
Training with on-the-fly random transformations ($p=0.8$) including JPEG ($Q \in [30, 90]$), Gaussian blur ($\sigma \in [0.5, 2.0]$), resizing ($0.25\times\text{--}0.5\times$), Gaussian noise ($\sigma \in [0.02, 0.10]$), jitter ($\pm 20\%$), and cropping elevated the unaugmented baseline's catastrophic noise collapse from **$60.71\% \rightarrow 94.53\%$** without hurting clean accuracy ($97.03\% \rightarrow 96.83\%$).

### 2. Why the FFT Branch Succeeded (Phase 4)
The 2D FFT CNN branch provided **82 unique correct classifications** where spatial RGB failed, delivering a net $+43$ sample ($+0.72\%$) advantage on the development validation split. The magnitude spectrum captures high-frequency periodicity from convolutional upsampling that spatial kernels miss.

### 3. Why Hybrid V4 Was Rejected
Hybrid V4 tested training-time loss reweighting (`pos_weight = 1.35`) to bias gradients toward synthetic recall. The heavier loss penalty caused premature convergence (Epoch 3) before spatial and spectral representations could balance. Consequently, V4 degraded across all 17 conditions: clean accuracy fell by $-1.02\%$ ($96.53\%$), mean transformed accuracy fell by $-0.92\%$ ($96.03\%$), and the worst-case floor fell to $94.75\%$.

### 4. Why the Deployment Threshold is Locked at $\theta = 0.50$
Canonical threshold calibration across 102,017 test passes proved that $\theta = 0.50$ maximizes the multi-condition robustness floor (**$95.55\%$** at `noise_0.10`) while keeping false positive alarms on authentic images strictly constrained at $2.50\%$ (clean) and $2.70\%$ (held-out COCO).

---

## Quickstart & CLI Usage

### Installation

Clone the repository and activate the environment:

```powershell
git clone https://github.com/AmitejSingh1/Verixa.git C:\Verixa
cd C:\Verixa

# Activate environment
conda activate verixa

# Install package in editable mode
pip install -e .
```

### 1. Single-Image Inference CLI

Run inference on any image using the locked Hybrid V1 champion model:

```powershell
python scripts/predict.py --image path/to/image.jpg
```

**Terminal Output Example:**
```text
=================================================================
 Verixa — AI-Generated Image Detection Inference CLI
=================================================================
 Model Checkpoint: models\convnext_tiny_hybrid_fft.pt
 Compute Device:   cuda
 Decision Cut-Off: 0.5000
=================================================================

Analysis Result:
  File:           path/to/image.jpg
  Classification: AUTHENTIC (Class 0)
  Probability:    0.0001 (Synthetic score)
  Confidence:     99.9%
  Inference Time: 18.4 ms
```

### 2. Batch Directory Inference

Process an entire folder of mixed images with GPU batch acceleration:

```powershell
python scripts/predict.py `
  --image-dir path/to/images/ `
  --batch-size 32 `
  --output predictions.csv
```

### 3. Python API Integration

```python
from pathlib import Path
from verixa.inference import VerixaPredictor

# Initialize predictor with locked champion checkpoint
predictor = VerixaPredictor(checkpoint_path="models/convnext_tiny_hybrid_fft.pt")

# Single image classification
result = predictor.predict_image("test_image.jpg")
print(f"Class: {result['class_name']} | Confidence: {result['confidence_pct']}%")

# Batch classification
batch_results = predictor.predict_batch([Path("img1.png"), Path("img2.jpg")])
predictor.save_predictions_to_csv(batch_results, "batch_output.csv")
```

---

## Complete Reproduction Pipeline

To reproduce all experimental results from raw data to final evaluation:

```powershell
# 1. Inspect datasets and generate manifest
python scripts/inspect_datasets.py --out reports/dataset_inspection.json

# 2. Ingest CIFAKE and SID_Set (30,000 processed images)
python scripts/ingest_local_dataset.py --root data/raw/cifake --source-dataset CIFAKE --limit-per-class 7500
python scripts/ingest_hf_dataset.py --dataset-id "competitions/sid_set" --source-dataset SID_Set --limit-per-class 7500

# 3. Deterministic deduplication and fixed split (seed 1337)
python scripts/detect_duplicates.py --manifest data/manifests/merged_manifest.csv
python scripts/create_fixed_split.py --manifest data/manifests/merged_manifest.csv --train-ratio 0.8 --seed 1337

# 4. Train Champion Hybrid V1
python scripts/train_hybrid.py `
  --manifest data/manifests/merged_manifest.csv `
  --aug-prob 0.8 `
  --epochs 20 `
  --batch-size 32 `
  --lr 0.0001 `
  --weight-decay 0.01 `
  --seed 1337 `
  --checkpoint-out models/convnext_tiny_hybrid_fft.pt

# 5. Evaluate complete 17-condition development robustness suite
python scripts/evaluate_robustness.py `
  --model-path models/convnext_tiny_hybrid_fft.pt `
  --manifest data/manifests/merged_manifest.csv `
  --report-out reports/final_hybrid_robustness_report.json

# 6. Run full unit test suite
pytest -v
ruff check .
```

---

## Repository Structure

```
C:\Verixa\
├── config/                     # Dataset schemas and label mapping configurations
├── data/
│   ├── manifests/              # Leakage-checked CSV manifests (merged_manifest.csv)
│   └── processed/              # Resized 224x224 JPEG images (gitignored)
├── models/
│   ├── convnext_tiny_hybrid_fft.pt      # PRIMARY CHAMPION CHECKPOINT (Epoch 6)
│   ├── convnext_tiny_robust_rgb.pt      # LOCKED FALLBACK CHECKPOINT (Epoch 3)
│   ├── convnext_tiny_hybrid_v2.pt       # Preserved negative result (Stage 2 unfreeze)
│   ├── convnext_tiny_hybrid_v3.pt       # Preserved negative result (LS + Flip)
│   └── convnext_tiny_hybrid_v4_weighted.pt # Preserved negative result (pos_weight=1.35)
├── reports/
│   ├── threshold_calibration.json       # Authoritative canonical threshold analysis
│   ├── error_analysis.md                # Phase 6 comprehensive forensic error report
│   ├── final_hybrid_robustness_report.json # 17-condition champion robustness metrics
│   └── held_out_benchmark_results.json  # Frozen held-out benchmark results
├── scripts/
│   ├── predict.py                       # Production CLI inference entrypoint
│   ├── generate_submission.py           # Competition submission generator
│   ├── evaluate_robustness.py           # 17-condition evaluation harness
│   ├── compute_canonical_thresholds.py  # Canonical threshold calibration tool
│   ├── train_hybrid.py                  # Hybrid V1 champion training script
│   └── train_rgb.py                     # Robust RGB training script
├── src/verixa/
│   ├── data/                           # Manifest schemas, hashing, and split logic
│   ├── evaluation/                     # AUROC, FPR@95% TPR, and calibration metrics
│   ├── inference.py                    # Production VerixaPredictor Python API
│   ├── models/
│   │   ├── convnext.py                 # ConvNeXt-Tiny wrapper & stage freeze logic
│   │   ├── fft.py                      # 2D FFT magnitude spectrum extractor
│   │   ├── hybrid.py                   # Dual-branch concatenation fusion model
│   │   └── loader.py                   # Universal checkpoint loader
│   └── training/
│       ├── augmentations.py            # Transformation-aware distortion augmentations
│       ├── dataset.py                  # PyTorch Dataset and DataLoader factories
│       └── trainer.py                  # Training engine with AMP, early stopping, VRAM logs
├── tests/                              # 68 automated unit tests (100% passing)
├── Phases.md                           # Detailed 8-phase project execution log
├── Rules.md                            # Engineering standards and hardware constraints
└── README.md                           # Master project documentation
```

---

## Limitations & Operational Envelope

1. **Unseen Modern Generator Shift:** While Verixa achieves a strong $92.13\%$ AUROC on unseen DALL·E 3 images, modern consistency decoders with high perceptual loss fine-tuning eliminate many traditional high-frequency artifacts. For investigative auditing prioritizing high synthetic recall, practitioners can inspect continuous probability outputs.
2. **Extreme Image Degradation:** Heavy additive Gaussian noise ($\sigma = 0.10$) reduces accuracy to $95.37\%$, and extreme downsampling ($0.25\times$) reduces accuracy to $96.27\%$ by obscuring fine spatial textures and frequency components.
3. **Hardware Efficiency:** Peak VRAM is capped at $\le 1.3\text{ GB}$ during training and $< 400\text{ MB}$ during inference, enabling rapid deployment on standard consumer GPUs and edge workstations ($< 20\text{ ms}$ per image on RTX 4060).
