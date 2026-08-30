# Verixa: Forensic AI-Generated Image Detection Under Real-World Transformations

**Devpost Project Submission Draft**  
*Track / Category:* AI Safety, Computer Vision, Digital Forensics  
*GitHub Repository:* [https://github.com/AmitejSingh1/Verixa](https://github.com/AmitejSingh1/Verixa)  

---

## 1. Problem & Inspiration

The rapid democratization of generative artificial intelligence (e.g., Latent Diffusion Models, Diffusion Transformers, Generative Adversarial Networks) has made high-fidelity synthetic image creation effortless. However, this has created acute risks: misinformation campaigns, identity impersonation, automated fraud, and non-consensual synthetic imagery.

While standard deep learning classifiers can easily detect pristine AI-generated images in lab environments, they suffer from **catastrophic robustness collapse** when deployed in the wild. Real-world images shared across social networks, messaging platforms, and content delivery networks undergo lossy JPEG compression, aggressive spatial downsampling, camera sensor noise, blurring, and color adjustments. Our unaugmented baseline experiments demonstrated that under severe Gaussian noise ($\sigma = 0.10$), a standard state-of-the-art vision model collapsed from $97.03\%$ accuracy down to **$60.71\%$**—barely better than random guessing.

**Verixa** was created to solve this vulnerability: an end-to-end, scientifically validated forensic detector engineered to maintain high classification reliability and low false alarm rates across clean and severely degraded imagery alike.

---

## 2. How Verixa Addresses the Problem

Verixa introduces a **Dual-Branch Hybrid Fusion Architecture** that attacks synthetic artifact detection simultaneously from two orthogonal domains:

1. **Spatial Domain Perception (ConvNeXt-Tiny):**
   A ConvNeXt-Tiny backbone pretrained on ImageNet-1K. Stages 0 through 2 are strictly frozen to preserve universal visual primitives, while Stage 3 and the classification head are fine-tuned to capture semantic anomalies, non-physical shading, plasticized textures, and anatomical incoherence.
2. **Spectral Domain Frequency Analysis (2D FFT CNN):**
   A specialized 4-stage convolutional neural network operating directly on centered 2D Fast Fourier Transform magnitude spectra ($\log(1 + |F(u, v)|)$). Generative models rely on upsampling convolutions or pixel shuffle layers that leave subtle periodic grid artifacts in the frequency domain. The spectral branch captures these periodic signatures even when spatial textures are partially obscured.
3. **Transformation-Aware Robust Training:**
   Models are trained with on-the-fly, randomized transformations ($p = 0.8$) simulating real-world pipeline degradation: JPEG compression ($Q \in [30, 90]$), Gaussian blur ($\sigma \in [0.5, 2.0]$), bilinear resizing ($0.25\times\text{--}0.5\times$), additive Gaussian noise ($\sigma \in [0.02, 0.10]$), color jitter ($\pm 20\%$), and cropping ($80\%$).

---

## 3. Development Tools Used

- **Integrated Development Environment (IDE):** Visual Studio Code with PowerShell terminal integration.
- **Hardware Platform:** Local development on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM). Peak memory usage was strictly engineered to remain $< 1.3\text{ GB}$ during training and $< 400\text{ MB}$ during inference.
- **Compute Stack:** NVIDIA CUDA 12.8, cuDNN, and PyTorch mixed-precision (`torch.amp` FP16) acceleration.
- **Environment & Version Control:** Anaconda isolated Python 3.11 environment, Git, and GitHub for version control.
- **Code Quality & Verification:** `ruff` for linting and formatting; `pytest` for automated test-driven development (68 automated unit tests).

---

## 4. Models & Architecture Specifications

- **Spatial Branch:** `torchvision.models.convnext_tiny(weights=IMAGENET1K_V1)`. Output feature dimension: $768$.
- **Spectral Branch:** Custom 4-stage 2D FFT CNN (`FFTSpectrumExtractor` + Conv blocks + AdaptiveAvgPool2d). Output feature dimension: $512$.
- **Concatenation Fusion Head:** Concatenates spatial ($768\text{-d}$) and spectral ($512\text{-d}$) features into a $1,280\text{-dimensional}$ vector $\rightarrow$ Linear($1280 \rightarrow 256$) $\rightarrow$ GELU $\rightarrow$ Dropout($p=0.3$) $\rightarrow$ Linear($256 \rightarrow 1$) $\rightarrow$ Sigmoid.
- **Parameter Efficiency:**
  - Total Parameters: $29,784,130$ (~29.8M parameters).
  - Trainable Parameters: $17,436,130$ (~17.4M parameters).
  - Frozen Backbone Parameters: $12,348,000$ (~12.3M parameters).
  - This utilizes $< 1.5\%$ of the competition's 2-Billion parameter ceiling.
- **Deployment Threshold:** Strictly locked at $\theta = \mathbf{0.50}$.

---

## 5. Libraries & Frameworks

- **Core Deep Learning:** `torch` (2.6.0+cu128), `torchvision` (0.21.0+cu128).
- **Computer Vision & Image Processing:** `Pillow` (PIL), `NumPy`.
- **Metrics & Data Science:** `scikit-learn` (AUROC, ROC curves, confusion matrices, quantiles).
- **Code Health & Testing:** `pytest`, `pytest-cov`, `ruff`.

---

## 6. Datasets & Assets

1. **Development Dataset (30,000 Images, 50/50 Balanced):**
   - **CIFAKE:** $15,000$ images ($7,500$ authentic CIFAR-10 / $7,500$ synthetic Stable Diffusion v1.4).
   - **SID_Set:** $15,000$ images ($7,500$ authentic photography / $7,500$ synthetic images from diverse diffusion and GAN engines). Tampered/edited images (`label=2`) were filtered out to preserve strict binary integrity.
   - **Standardization:** Resized to $224 \times 224$ bicubic, stored as compressed JPEGs ($Q=90$, total size: $307.89\text{ MB}$).
   - **Deterministic Split:** Assigned using global seed `1337` and SHA-256 group hashing to produce $23,999$ training samples and $6,001$ validation samples with **strictly zero cross-split leakage**.
2. **Held-Out Benchmark (13,841 Images):**
   - **Authentic:** $4,998$ images from Microsoft COCO val2017.
   - **Synthetic:** $8,843$ high-resolution images generated by OpenAI DALL·E 3.
   - Kept under strict quarantine throughout model selection and hyperparameter exploration.

---

## 7. Development Process & Experimental Journey

Verixa was developed across an 8-phase milestone roadmap with strict scientific gates:

- **Phase 1 (Data Engineering):** Ingestion, perceptual hash deduplication, leakage verification, and dataset freeze.
- **Phase 2 (Clean Baseline):** Clean ConvNeXt-Tiny achieved $97.03\%$ clean accuracy, but collapsed to $60.71\%$ on severe noise.
- **Phase 3 (Transformation-Aware Robust RGB):** Training with $p=0.8$ augmentations elevated noise accuracy to $94.53\%$ (+33.82 points) and worst-case floor to $94.33\%$. Locked as the official **Fallback Model**.
- **Phase 4 (FFT Exploration & Hybrid Fusion):** Standalone FFT verified frequency discriminability ($84.24\%$ clean acc, $94.65\%$ AUROC). The Hybrid RGB+FFT V1 outperformed the pure RGB model across **16 out of 16 conditions**, lifting clean accuracy to $97.55\%$, mean transformed accuracy to $96.95\%$, and worst-case floor to $95.37\%$. Designated as the **Champion Model**.
- **Post-Phase 5 Controlled Ablation Experiments:**
  - *Threshold Calibration:* Evaluated across 102,017 test passes; confirmed deployment threshold $\theta = 0.50$ is optimal for preserving worst-case robustness floor ($95.55\%$).
  - *Hybrid V2 (Stage 2 Unfreezing):* Degraded noise floor to $93.75\%$. **Rejected.**
  - *Hybrid V3 (Label Smoothing + Horizontal Flip):* Degraded clean accuracy to $96.80\%$ and clean FPR to $3.77\%$. Horizontal flips disrupted 2D FFT directional magnitude spectrum orientation. **Rejected.**
  - *Hybrid V4 (Weighted Loss, `pos_weight=1.35`):* Authoritative results: Clean Acc $96.53\%$, Clean FPR $2.07\%$, Mean Trans Acc $96.03\%$, Worst-Case Floor $94.75\%$. Failed promotion gates. **Rejected.**
- **Phase 6 (Forensic Error Analysis):** In-depth study of failure modes and spatial/spectral complementarity.
- **Phase 7 (Inference CLI):** Production-grade Python API (`VerixaPredictor`) and CLI (`scripts/predict.py`).
- **Phase 8 (Documentation & Verification):** Full documentation, 68 passing unit tests, clean linting.

---

## 8. Robustness & Benchmark Results

### 17-Condition Development Suite ($N = 6,001$)

| Condition | Baseline ConvNeXt | Robust RGB (Fallback) | Hybrid RGB+FFT (Champion) | Net Gain (Champion vs Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Validation** | 97.03% | 96.83% | **97.55%** | $+0.52\%$ |
| JPEG Quality 90 | 96.88% | 96.82% | **97.53%** | $+0.65\%$ |
| JPEG Quality 70 | 96.38% | 96.77% | **97.38%** | $+1.00\%$ |
| JPEG Quality 50 | 95.82% | 96.55% | **97.22%** | $+1.40\%$ |
| JPEG Quality 30 (Severe) | 92.90% | 96.00% | **96.77%** | $+3.87\%$ |
| Gaussian Blur $\sigma=0.5$ | 96.87% | 96.80% | **97.48%** | $+0.61\%$ |
| Gaussian Blur $\sigma=1.0$ | 96.42% | 96.70% | **97.38%** | $+0.96\%$ |
| Gaussian Blur $\sigma=2.0$ (Severe) | 92.67% | 95.23% | **96.68%** | $+4.01\%$ |
| Resize Scale $0.50\times$ | 96.03% | 96.65% | **97.27%** | $+1.24\%$ |
| Resize Scale $0.25\times$ (Severe) | 88.19% | 94.33% | **96.27%** | $+8.08\%$ |
| Gaussian Noise $\sigma=0.02$ | 87.64% | 96.63% | **97.18%** | $+9.54\%$ |
| Gaussian Noise $\sigma=0.05$ | 69.84% | 95.87% | **96.38%** | $+26.54\%$ |
| Gaussian Noise $\sigma=0.10$ (Severe) | 60.71% | 94.53% | **95.37%** | **$+34.66\%$** |
| Color Jitter $\pm 10\%$ | 96.88% | 96.85% | **97.47%** | $+0.59\%$ |
| Color Jitter $\pm 20\%$ (Severe) | 96.72% | 96.80% | **96.98%** | $+0.26\%$ |
| Center Crop 80% | 95.73% | 96.60% | **96.72%** | $+0.99\%$ |
| **Composite Severe Transformation** | 89.12% | 96.33% | **97.05%** | $+7.93\%$ |
| **Mean Transformed Accuracy** | 88.34% | 96.11% | **96.95%** | **$+8.61\%$** |
| **Worst-Case Floor** | 60.71% | 94.33% | **95.37%** | **$+34.66\%$** |

### Held-Out Benchmark Performance ($N = 13,841$, Unseen COCO + DALL·E 3)

| Model | AUROC | Overall Acc | COCO Acc (Real) | COCO FPR (False Alarms) | DALL·E 3 Recall (Fake) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RGB+FFT (Champion)** | **92.13%** | **68.34%** | **97.30%** | **2.70%** | **51.97%** |
| **Robust RGB (Fallback)** | 90.74% | 66.79% | 94.36% | 5.64% | 51.20% |
| **Net Gain ($\Delta$)** | **+1.39%** | **+1.55%** | **+2.94%** | **-2.94%** | **+0.77%** |

- **52.1% Reduction in False Alarms:** On authentic COCO photographs, the Hybrid champion halved false accusations ($5.64\% \rightarrow 2.70\%$) compared to pure RGB.
- **Superior Discriminative Ranking:** Held-out AUROC reached **$92.13\%$**, proving the model ranks subtle fakes higher than authentic photos in $>92\%$ of cases.

---

## 9. Limitations & Challenges

1. **Latent Diffusion Transformer Decoders:** DALL·E 3 images use proprietary latent decoders that minimize high-frequency upsampling noise. Because the training set lacked DALL·E 3 samples, zero-shot sensitivity is lower ($51.97\%$) at $\theta = 0.50$ than on seen training generators ($99.3\%$).
2. **Extreme Compounded Distortions:** Heavy additive Gaussian noise ($\sigma = 0.10$) and downsampling ($0.25\times$) degrade detection confidence by partially masking subtle pixel and frequency signatures.
3. **High-Contrast Lighting False Positives:** Direct flash photography and extreme concert lighting occasionally mimic diffusion gradients.

---

## 10. What We Would Improve With More Time

1. **Self-Supervised Pretraining on Diffusion Latent Spaces:** Train an auxiliary self-supervised encoder directly on the latent spaces of modern consistency models (e.g., SDXL, Flux, Midjourney v6) to learn decoder residuals invariant to content.
2. **Multi-Scale Directional Spectral Pooling:** Replace global 2D FFT pooling with steerable pyramid or wavelet decomposition to preserve rotational and directional frequency signatures during augmentation.
3. **Adaptive Test-Time Calibration:** Incorporate unsupervised image quality estimation (e.g., blind JPEG quality factor and noise variance estimation) to dynamically calibrate decision boundaries per image.
4. **Ensemble Distillation:** Distill the joint predictions of multiple complementary backbones (e.g., Swin-Transformer + ConvNeXt + EfficientNet) into a lightweight single-pass model.
