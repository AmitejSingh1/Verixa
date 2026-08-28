# Product Requirements Document (PRD) — Verixa

## 1. Project Overview & Problem Statement

**Project Name:** Verixa  
**Project Focus:** Robust AI-Generated Image Detection Under Real-World Transformations  
**Competition Context:** TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images Under Real-World Transformations  
**Repository Location:** `C:\Verixa`  

### 1.1 Problem Statement
Generative AI models (Diffusion Models, GANs, Autoregressive Models) can synthesize photorealistic imagery capable of bypassing conventional human inspection and naive detection systems. However, in real-world platform distribution (e.g., social media uploads, messaging apps, content management systems), images rarely remain in pristine, uncompressed form. They undergo social-media re-compression, resizing, noise corruption, color adjustments, blurring, and cropping.

Standard deepfake detectors frequently overfit to high-frequency artifacts (such as checkerboard patterns or frequency grid peaks) present only in uncompressed images. When subjected to lossy transformations—especially JPEG compression or downsampling—these fragile artifacts are destroyed, causing naive detectors to suffer catastrophic performance collapse.

### 1.2 Core Objective
The core objective of Verixa is **not** merely maximizing clean-image classification accuracy. The primary technical objective is **robustness**: distinguishing AI-generated images from authentic images reliably even after images have undergone realistic post-processing and distortion pipelines.

Verixa produces a calibrated probability score indicating the likelihood that an input image is AI-generated ($P(\text{AI-Generated}) \in [0.0, 1.0]$).

---

## 2. Target Users & Use Cases

Verixa is designed as a modular, high-throughput detection component for:
1. **Trust & Safety Platforms:** Automated screening pipelines assessing user uploads for undeclared synthetic media before distribution.
2. **Content Moderation & Verification Workflows:** Assisting human moderators with calibrated confidence scores and forensic signals across multi-stage social platforms.
3. **Digital Media Auditing & Fact-Checking:** Verifying imagery in news, media authentication, and verification environments where images are received after unknown rounds of compression and web transfer.

> [!IMPORTANT]
> **Authenticity Scope:** Verixa is a probabilistic binary classifier estimating AI generation likelihood. It does **not** claim to provide cryptographically unfalsifiable proof of provenance or absolute certainty.

---

## 3. Scope & Non-Goals

### 3.1 In-Scope
- Binary classification: Class `0` (REAL / Authentic) vs. Class `1` (AI-GENERATED / Synthetic).
- Ingestion and normalization of multi-source training datasets (SID_Set, CIFAKE, WildFake).
- Transformation-aware training using randomized CPU-side data augmentation in the dataloader.
- Evaluation across 6 required real-world transformation families and multiple severity levels.
- Pretrained **ConvNeXt-Tiny** backbone fine-tuned within an 8 GB VRAM hardware budget.
- Controlled exploration of an experimental dual-branch architecture (RGB ConvNeXt-Tiny + FFT frequency branch), retained only if experimentally justified.
- Deterministic, leakage-free train/validation splitting with cryptographic hash checks.
- Comprehensive evaluation reporting Accuracy, AUROC, False Positive Rate (FPR), and Robustness Degradation.
- Evaluation on an isolated held-out benchmark (COCO val2017 authentic + DALL-E Advanced synthetic).

### 3.2 Non-Goals
- **No Inpainting/Tampering Localization:** Detecting localized pixel tampering or segmenting edited regions is out of scope. SID_Set label `2` (tampered) is explicitly excluded from binary training.
- **No Video Processing:** Verixa processes static 2D images only.
- **No Full Dataset Ingestion:** Downloading massive full-scale source archives locally is strictly prohibited due to local storage limits.
- **No Giant Models / Ensembles:** Model parameter count must remain strictly below the competition limit of 2 Billion parameters (ConvNeXt-Tiny has ~28M parameters). Giant multi-model ensembles are out of scope unless experimentally necessary and resource-compliant.
- **No Pre-generated Augmented Datasets:** Distortions are computed on-the-fly to conserve disk space.

---

## 4. Dataset Specifications

### 4.1 Training & Validation Sources

| Dataset | Provider / Source | Intended Role | Label Mapping & Status |
| :--- | :--- | :--- | :--- |
| **SID_Set** | Hugging Face (`saberzl/SID_Set`) | Diverse authentic photos & modern full-synthetic images | `0` $\rightarrow$ **0 (REAL)**<br>`1` $\rightarrow$ **1 (AI-GENERATED)**<br>`2` $\rightarrow$ **EXCLUDE (Tampered)**<br>*Status: 15,000 images in final dataset* |
| **CIFAKE** | Kaggle (`birdy654/cifake-real-and-ai-generated-synthetic-images`) | Balanced baseline volume (CIFAR-10 real vs SD 1.4 fake) | `REAL` $\rightarrow$ **0 (REAL)**<br>`FAKE` $\rightarrow$ **1 (AI-GENERATED)**<br>*Status: 15,000 images in final dataset* |
| **WildFake** | ModelScope (`hy2628982280/WildFake`) | Multi-generator & architectural diversity | **EXCLUDED FROM TRAINING**<br>*Rationale: Stored exclusively as 16–51 GB monolithic zip archives on ModelScope; cannot be streamed or selectively sampled without exceeding local storage budgets.* |

### 4.2 Held-Out Benchmark (Strictly Isolated)
- **Authentic (non-AIGC):** COCO val2017
- **AI-Generated (AIGC):** DALL-E Advanced
- **Constraint:** The held-out benchmark must **never** be used for model training, validation tuning, threshold calibration, or model selection during development. It is reserved exclusively for the final evaluation phase.

### 4.3 Ingestion & Dataset Sizing Strategy
- **Pilot Dataset (Completed):** 8,000 standardized images (4,000 CIFAKE + 4,000 SID_Set; 6,401 train / 1,599 val) taking 91.3 MB on disk. Used strictly in Phases 1 & 2 to validate pipeline, AMP, VRAM stability, and metrics. Historical results permanently preserved.
- **Final Frozen Training Dataset:** Exactly **30,000 standardized images** (15,000 CIFAKE + 15,000 SID_Set; 24,000 train / 6,000 val) taking $\approx 345\text{ MB}$ on disk. Once generated, this split is frozen and used identically for both the official Phase 2 clean baseline and the Phase 3 robust model.
- **Standardization:** All retained images are converted to 3-channel RGB, bicubic resized to $224 \times 224$, and stored as JPEG ($Q=90$). Full-resolution raw images are never retained locally.

---

## 5. Required Robustness Transformations

Models must be evaluated on clean inputs and against the competition-mandated transformations:

```text
1. JPEG Compression:    quality ∈ {90, 70, 50, 30}
2. Gaussian Blur:       sigma ∈ {0.5, 1.0, 2.0}
3. Resize (Down/Up):    scale ∈ {0.5×, 0.25×} (bilinear downscale then upscale to 224×224)
4. Gaussian Noise:      sigma ∈ {0.02, 0.05, 0.10}
5. Color Jitter:        brightness / contrast / saturation ∈ [−20%, +20%]
6. Center Crop:         crop 80% of image dimensions, resized back to 224×224
```

---

## 6. Machine Learning & Modeling Requirements

### 6.1 Model Architecture
- **Primary Backbone:** Pretrained **ConvNeXt-Tiny** (~28.6M parameters). Fine-tuned with a binary classification head (Logits $\rightarrow$ Sigmoid / Binary Cross-Entropy with Logits).
- **Experimental Model:** Dual-Branch RGB ConvNeXt-Tiny + Fast Fourier Transform (FFT) Frequency Analysis Branch.
  - **Condition:** Explored strictly as an experimental branch in Phase 4. Retained in the final submission **only** if it demonstrates statistically meaningful robustness improvements over the RGB baseline.

### 6.2 Training Methodology
- **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **Precision:** Mixed Precision (`torch.amp.autocast(device_type='cuda', dtype=torch.float16)`).
- **Optimization:** AdamW optimizer with cosine annealing schedule and linear warmup.
- **Batch Size:** 16 to 32 (with gradient accumulation if required to stabilize updates).
- **Freezing Strategy:** Freeze early stages of ConvNeXt-Tiny initially; fine-tune stage 3/4 and classification head. Progressively unfreeze earlier layers only if underfitting is observed.
- **Augmentation Scheme:** Transformation-aware training applying randomly sampled subsets of the required transformations with randomized severity on CPU in the PyTorch `DataLoader`.

---

## 7. Evaluation & Success Criteria

### 7.1 Metrics
For every evaluation run (clean validation, individual transformation severities, and held-out benchmark), the following metrics must be computed and reported:
1. **Accuracy:** Fraction of correct predictions at default threshold ($0.5$).
2. **AUROC (Area Under the Receiver Operating Characteristic):** Threshold-independent discriminative performance.
3. **False Positive Rate (FPR):** Fraction of authentic images mistakenly flagged as AI-generated at fixed decision thresholds (e.g., at 95% True Positive Rate / Recall).
4. **Robustness Degradation ($\Delta_{\text{degradation}}$):** Drop in metric values from clean baseline to transformed conditions:
   $$\Delta_{\text{metric}} = \text{Metric}_{\text{clean}} - \text{Metric}_{\text{transformed}}$$

### 7.2 Target Success Criteria
- Clean Validation Accuracy: $\ge 92.0\%$ / AUROC $\ge 0.960$.
- Severe JPEG ($Q=30$) Accuracy: $\ge 80.0\%$ / Degradation $\le 12.0\%$.
- Severe Gaussian Blur ($\sigma=2.0$) Accuracy: $\ge 80.0\%$.
- False Positive Rate on authentic imagery: $\le 5.0\%$ at operational operating point.
- Inference latency: $< 50\text{ ms}$ per $224 \times 224$ image on RTX 4060 Laptop GPU.

---

## 8. Hardware & Resource Constraints

- **Compute:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM).
- **Host OS:** Windows (PowerShell environment).
- **VRAM Constraint:** Peak memory must stay below 7.2 GB during training to prevent Out-Of-Memory (OOM) faults.
- **Disk Budget:** Processed dataset storage must remain strictly budgeted ($< 500\text{ MB}$ for pilot; $< 2.5\text{ GB}$ if scaled to 50K).
- **Determinism:** Seed `1337` fixed across PyTorch, NumPy, Python standard library, and DataLoader workers.

