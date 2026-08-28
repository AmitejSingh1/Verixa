# System Architecture — Verixa

Verixa is an end-to-end deep learning framework designed to detect AI-generated imagery and maintain detection reliability under severe real-world transformations (JPEG compression, blur, resizing, noise, color jitter, and cropping).

---

## 1. High-Level System Architecture

```text
                                  SOURCE DATASETS
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
        CIFAKE (15,000)                                   SID_Set (15,000)
    (100% Local KaggleHub cache)                      (Hugging Face Streaming)
    7,500 REAL / 7,500 FAKE                           7,500 REAL / 7,500 AI-GEN
                │                                                 │
                └────────────────────────┬────────────────────────┘
                                         ▼
                             STANDARDIZATION PIPELINE
                             - 3-channel RGB conversion
                             - Bicubic resize to 224×224
                             - JPEG compression (Q=90)
                             - SHA-256 & Perceptual Hashing
                                         │
                                         ▼
                            FINAL FROZEN MANIFEST CSV
                        (30,000 rows — data/manifests/merged_manifest.csv)
                                         │
                        Deterministic Split (Seed 1337)
                        24,000 Train (80%) / 6,000 Val (20%)
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   CLEAN TRAINING PIPELINE                               ROBUST TRAINING PIPELINE (Phase 3)
   - Standard ImageNet Normalization                     - On-The-Fly CPU Augmentation Engine
   - No distortions                                      - JPEG Q in [30, 90], Blur, Noise, Crop
             │                                                       │
             ▼                                                       ▼
   ConvNeXt-Tiny Baseline                                 ConvNeXt-Tiny Robust Model
   (models/convnext_tiny_baseline.pt)                     (models/convnext_tiny_robust_rgb.pt)
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         ▼
                            HEAD-TO-HEAD BENCHMARKING
                            - Clean Validation (6,000 images)
                            - 6 Distortion Suites (18 severity levels)
                            - Quantify Delta_degradation reduction
                                         │
                                         ▼
                          FINAL OUT-OF-DISTRIBUTION BENCHMARK
                          - COCO val2017 (Authentic) + DALL-E Advanced (Synthetic)
                          - Strictly isolated until Phase 5
```

---

## 2. Dataset Architecture & Storage Strategy

### 2.1 Dataset Ingestion & Sizing Strategy
1. **Pilot Dataset (Historical Validation):** 8,000 images (4,000 CIFAKE + 4,000 SID_Set) taking 91.3 MB on disk. Successfully validated pipeline, AMP stability, peak VRAM (< 550 MB), and metrics. Checkpoint and reports permanently preserved (`models/convnext_tiny_baseline_8k_pilot.pt`).
2. **Final Frozen Dataset:** Exactly 30,000 images (15,000 CIFAKE + 15,000 SID_Set; 24,000 train / 6,000 val) taking $\approx 345\text{ MB}$ on disk.
3. **WildFake Assessment:** WildFake (`hy2628982280/WildFake` on ModelScope) stores images strictly inside monolithic `.zip` archives per architecture (e.g. `GAN_based.zip` 45.1 GB, `DALLE.zip` 24.4 GB, `ADM.zip` 17.6 GB, `Midjourney.zip` 51.1 GB). Because direct HTTP streaming to individual images returns 404, sampling requires downloading hundreds of gigabytes, which directly violates storage limits. WildFake is therefore excluded from the training dataset.
4. **Held-Out Benchmark Isolation:** COCO val2017 (authentic) and DALL-E Advanced (synthetic) are stored in `data/held_out_benchmark/` and never accessed during training, tuning, or early stopping.

### 2.2 Standard Manifest Schema (`merged_manifest.csv`)
All dataset records adhere to a strict 7-column schema:
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_path` | String | Relative or absolute path to the local 224×224 JPEG file |
| `label` | Integer | Binary classification target: `0` = REAL, `1` = AI-GENERATED |
| `source_dataset` | String | Name of the originating dataset (`CIFAKE` or `SID_Set`) |
| `original_id` | String | Unique identifier or filename from the source repository |
| `generator` | String | Generator architecture metadata where verified, else empty |
| `sha256` | String | 64-character hexadecimal SHA-256 hash of the normalized JPEG file |
| `split` | String | Assigned partition: `train` or `val` |

---

## 3. Model Architecture

### 3.1 Primary Model: ConvNeXt-Tiny Binary Classifier
- **Backbone:** Torchvision's `convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)`.
- **Parameter Count:** ~28.6M total parameters (well below competition 2B parameter ceiling).
- **Freezing Scheme:** Stages 0, 1, and 2 are frozen during fine-tuning (~13.0M params frozen). Stage 3 and the classification head remain active (~15.7M trainable params).
- **Classification Head:**
  $$\mathbf{x} \in \mathbb{R}^{B \times 768 \times 7 \times 7} \xrightarrow{\text{AdaptiveAvgPool2d}} \mathbb{R}^{B \times 768} \xrightarrow{\text{LayerNorm}} \mathbb{R}^{B \times 768} \xrightarrow{\text{Linear}} \mathbb{R}^{B \times 256} \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.3)} \xrightarrow{\text{Linear}} \mathbb{R}^{B \times 1}$$
- **Output:** Raw scalar logit per image for numerically stable `BCEWithLogitsLoss`.

---

## 4. Hardware Budgets & Execution Constraints
- **GPU Target:** NVIDIA GeForce RTX 4060 Laptop GPU (8,192 MB VRAM).
- **VRAM Hard Ceiling:** Peak allocated memory must remain $< 6.0\text{ GB}$ (measured baseline peak: ~546 MB).
- **Precision:** Mixed Precision (`torch.amp.autocast(dtype=torch.float16)`) with `GradScaler`.
- **Disk Storage:** Processed image storage capped at $< 500\text{ MB}$ (30K dataset utilizes $\approx 345\text{ MB}$).
- **Random Seed:** Globally fixed to `1337` across PyTorch, NumPy, Python standard library, and CuDNN.

