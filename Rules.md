# Engineering Rules & Agent Guidelines — Verixa

> [!IMPORTANT]
> This is the foundational rulebook for all human developers and AI coding agents operating in the Verixa repository (`C:\Verixa`). Every instruction in this document is strict and binding.

---

## 1. General Workflow & Development Rules

### 1.1 Strict Phase Sequencing

Follow the eight development phases defined in `Phases.md` sequentially.

Do **not**:

* skip phases
* jump ahead
* implement components from later phases out of order
* begin FFT/hybrid work before the required RGB robustness checkpoint
* begin final benchmark evaluation before the model-selection process is complete

---

### 1.2 Atomic Phase Commits

Never combine multiple phases into a single Git commit.

Each phase must be committed individually using its designated commit message once its completion criteria are met.

---

### 1.3 Repository Runnable at All Times

Every commit and phase transition must leave the repository in a working state.

Before marking a phase complete:

```text
pytest -v
ruff check .
```

must pass cleanly unless a documented, approved exception exists.

---

### 1.4 Test Before Progressing

Do not proceed to the next phase until:

1. the current phase's implementation is complete
2. its required tests pass
3. the linter passes
4. required outputs/reports exist
5. the results have been reviewed

---

### 1.5 No Invented Requirements

Do not add:

* unapproved features
* datasets
* external services
* model architectures
* complex abstractions
* evaluation procedures

unless they are explicitly agreed upon.

If an additional idea appears useful, propose it first rather than silently implementing it.

---

### 1.6 No Silent Changes

Never silently change previously agreed:

* datasets
* dataset sizes
* labels
* random seeds
* train/validation splits
* thresholds
* transformations
* model architecture
* training hyperparameters
* evaluation methodology
* storage limits
* hardware constraints

If a change is necessary, explain the reason and obtain user approval before making the consequential change.

---

## 2. User-Controlled Execution & Command Transparency

### 2.1 User Must See Important Commands

Whenever an important command needs to be executed locally, the AI coding agent must provide the exact command to the user so the user can run it in their own IDE terminal.

The user should be able to observe:

* program output
* training progress
* GPU usage
* VRAM usage
* warnings
* errors
* generated files
* experiment results

---

### 2.2 Required Execution Workflow

For important operations, follow:

```text
AI proposes implementation
        ↓
AI explains what the code does
        ↓
AI provides exact terminal command
        ↓
USER runs command locally
        ↓
USER reviews output
        ↓
USER provides output when analysis is needed
        ↓
AI analyzes result
        ↓
AI proposes next step
```

---

### 2.3 Commands Must Be Explicit

Do not say only:

> "Run the training script."

Instead provide the complete command, for example:

```powershell
conda activate verixa
python scripts/train_rgb.py --epochs 1 --batch-size 32
```

Also explain:

1. what the command does
2. which script/file it executes
3. what output should appear
4. where outputs will be saved
5. relevant GPU/VRAM/storage considerations

---

### 2.4 Important Operations Must Not Be Hidden

Do not silently execute or initiate:

* long model training
* large dataset downloads
* deletion of datasets
* deletion of checkpoints
* overwriting experiment results
* changes to the main dataset
* major architecture changes
* major hyperparameter experiments

The user must be given the relevant command and understand what it will do.

---

### 2.5 Prefer One Important Command at a Time

For major experiments:

```text
1. Provide command.
2. User runs it.
3. Review result.
4. Provide next command.
```

Do not provide a long chain of commands that could cause later steps to execute on top of an unsuccessful earlier step.

---

### 2.6 Training Visibility

Training scripts should expose useful progress information, including where practical:

* epoch
* training loss
* validation loss
* learning rate
* Accuracy
* AUROC
* FPR
* GPU memory usage
* checkpoint status

Long-running training should never be treated as a black box.

---

## 3. Dataset & Storage Rules

### 3.1 Pilot vs Main Dataset

Verixa distinguishes between the **pilot dataset** and the **main experimental dataset**.

### Pilot Dataset

The current Phase 1 pilot contains:

```text
CIFAKE:
    2,000 REAL
    2,000 AI

SID_Set:
    2,000 REAL
    2,000 AI

Total:
    8,000 images
```

The pilot exists primarily to validate:

* the data pipeline
* image loading
* labels
* manifests
* splitting
* model training mechanics
* GPU/VRAM behavior
* evaluation code

A short pilot training run is an **engineering smoke test**.

Its metrics must **not** be presented as the official baseline if a larger main dataset is subsequently created.

### Main Experimental Dataset

Before official model-comparison experiments, create a larger, practical training dataset where possible, targeting approximately:

**30,000–50,000 images**

The exact size must be based on:

* accessible data
* dataset diversity
* storage limits
* compute constraints
* data quality

Once the main dataset is finalized and split, it must be **frozen** for model-comparison experiments.

---

### 3.2 Same Dataset for Fair Model Comparisons

All official baseline, robustness, and architecture-comparison experiments must use the **same frozen training and validation dataset**.

For example:

```text
Official RGB Baseline
        ↓
Same training set
Same validation set
        ↓
Robust RGB Model
        ↓
Same training set
Same validation set
        ↓
RGB + FFT Experiment
```

Do not change dataset size or composition between compared models unless the experiment is explicitly measuring the effect of dataset size.

---

### 3.3 No Bulk Source Downloads

Never attempt to download entire multi-gigabyte source archives unnecessarily.

Use:

* streaming
* controlled subsampling
* supported selective-download mechanisms

whenever practical.

Do not download a massive source archive merely to obtain a small number of images if doing so violates the project's storage/compute constraints.

---

### 3.4 WildFake Access Constraint

WildFake is an intended data source because of its generator diversity.

However, its current ModelScope distribution is stored in very large archive files.

Do **not** download entire multi-gigabyte WildFake archives solely to obtain a small controlled subset when this violates the project's storage constraints.

The current Phase 1 pilot therefore uses:

```text
CIFAKE + SID_Set
```

WildFake may be added later only if a practical and storage-compliant access method is found.

If WildFake is added, document:

* exact source
* sampling method
* generator/architecture metadata
* number of images
* storage usage
* any changes to the frozen main dataset

---

### 3.5 Held-Out WildFake Benchmark Isolation

The competition-provided held-out benchmark must remain completely separate from training data.

The benchmark consists of:

```text
COCO val2017:
    4,998 authentic images

DALL·E Advanced:
    8,843 AI-generated images
```

These images must **never** be used for:

* training
* validation
* augmentation development
* hyperparameter tuning
* threshold selection
* architecture selection
* intermediate model comparison

They are reserved for final evaluation.

---

### 3.6 Storage Limits

Pilot dataset:

**< 500 MB**

Main dataset target:

**< 2.5 GB**

If a proposed operation risks exceeding these limits, stop and report the expected/actual numbers before proceeding.

---

### 3.7 Standardize Immediately on Ingestion

All retained training images must be:

1. converted to 3-channel RGB
2. resized to `224 × 224`
3. saved as compressed JPEG, quality `Q=90`

Do not retain unnecessary full-resolution source images locally.

---

### 3.8 Preserve Dataset Provenance

Every manifest row must preserve provenance information whenever available.

At minimum:

```text
image_path
label
source_dataset
original_id
sha256
```

When available, also record:

```text
source_category
generator
architecture
```

---

### 3.9 No Invented Generator Metadata

If a dataset provides generator/architecture metadata, record it faithfully.

If the dataset does not provide specific generator information, do not invent it.

For example:

```text
SID_Set:
    generator = full_synthetic
```

is acceptable.

Inventing:

```text
SID_Set:
    generator = Stable Diffusion
```

when the dataset does not provide that information is prohibited.

For WildFake, architecture metadata should be recorded as the provided architecture/generator-family information.

---

### 3.10 Dataset Labels

The common Verixa label convention is:

```text
0 = REAL
1 = AI-GENERATED
```

For SID_Set:

```text
0 = real          → 0
1 = full_synthetic → 1
2 = tampered      → EXCLUDE
```

Do not silently change this mapping.

---

### 3.11 Mandatory Leakage & Deduplication Checks

Before training on any finalized manifest:

* run exact SHA-256 duplicate detection
* run perceptual near-duplicate analysis where applicable
* prevent exact duplicates from crossing train/validation splits
* verify zero cross-split hash leakage

Generator/source information should also be analyzed where available.

---

## 4. Machine Learning & Modeling Rules

### 4.1 ConvNeXt-Tiny is the Primary Model

The primary and fallback-ready architecture is:

**ImageNet-pretrained ConvNeXt-Tiny**

with RGB input.

It is the required starting architecture for the main experiments.

---

### 4.2 No Training From Scratch

Vision backbones must use official pretrained weights.

The primary configuration uses ImageNet-1K pretrained ConvNeXt-Tiny.

Training a large vision backbone from random initialization is prohibited.

---

### 4.3 Binary Classification

The detector must use:

```text
0 = REAL
1 = AI-GENERATED
```

A single-logit binary classifier with `BCEWithLogitsLoss` is the default approach unless explicitly changed.

---

### 4.4 Initial ConvNeXt Fine-Tuning Strategy

The initial configuration should:

* freeze early ConvNeXt stages
* train the deeper stage and classification head

If evidence of underfitting appears, deeper layers may be progressively unfrozen.

Do not perform unnecessary architecture/freeze experiments without a clear experimental reason.

---

### 4.5 FFT is Strictly Experimental

FFT is a hypothesis, not a guaranteed component of the final system.

The proposed experimental architecture is:

```text
                 IMAGE
                /     \
               ↓       ↓
             RGB      FFT
               ↓       ↓
          ConvNeXt    CNN
               \       /
                ↓     ↓
                 Fusion
                   ↓
              Classification
```

The FFT branch must only be adopted if it demonstrates a meaningful improvement over the robust RGB model on the predefined evaluation suite.

Do not assume FFT will improve robustness.

In particular, remember that JPEG compression and Gaussian blur can remove high-frequency information, which may weaken frequency-based features.

---

### 4.6 No Training From Scratch for FFT Backbone

The FFT branch should remain small and computationally practical.

Do not introduce a large second backbone or unnecessary architectural complexity.

---

### 4.7 No Unjustified Ensembles

The primary objective is a single robust detector.

Do not build a large ensemble unless an experiment demonstrates that the additional complexity is justified.

---

### 4.8 Parameter Limit

The complete final model must remain strictly below:

**2 billion parameters**

The parameter count must be measured and recorded rather than assumed.

ConvNeXt-Tiny is approximately 28.6M parameters and is comfortably within the limit.

---

### 4.9 Reproducibility

Use fixed seed:

```text
1337
```

where applicable across:

* Python `random`
* NumPy
* PyTorch
* CUDA
* DataLoader workers
* sampling procedures

Record the seed in experiment configurations and reports.

---

### 4.10 Hardware Constraints

Training must use the local:

**NVIDIA RTX 4060 Laptop GPU — 8 GB VRAM**

Use:

* mixed precision
* `torch.amp.autocast`
* batch size between 16 and 32 where practical
* gradient accumulation if necessary
* CPU-side transformations

If an experiment risks exceeding approximately **7.2 GB VRAM** or produces an OOM error:

**stop and report the actual memory situation.**

Do not silently change the experiment's core configuration.

---

## 5. Training Rules

### 5.1 Phase 2 Must Be Clean RGB

The official RGB baseline must use clean images without transformation-aware robustness augmentation.

Do not add:

* JPEG augmentation
* blur
* resize degradation
* Gaussian noise
* color jitter
* center crop augmentation
* FFT

to the official clean baseline.

---

### 5.2 Phase 3 Uses Transformation-Aware Training

The robust RGB model should use the competition's transformation families during training.

Training transformations must be sampled rather than applying every maximum severity to every image.

---

### 5.3 Evaluation Transformations Must Match the Specification

The evaluation suite must use exactly:

#### JPEG Compression

```text
Q = 90, 70, 50, 30
```

#### Gaussian Blur

```text
σ = 0.5, 1.0, 2.0
```

#### Resize

```text
0.5× → upscale
0.25× → upscale
```

#### Gaussian Noise

```text
σ = 0.02, 0.05, 0.10
```

#### Color Jitter

```text
brightness/contrast/saturation ±20%
```

#### Center Crop

```text
80% crop
```

Do not silently change these evaluation settings.

---

## 6. Evaluation & Reporting Rules

### 6.1 Zero Cherry-Picking

Report complete evaluation results.

Do not remove:

* difficult classes
* difficult severities
* poor-performing transformations
* inconvenient examples

because they make the model look worse.

---

### 6.2 Fixed Validation Set

All official model comparisons must use the exact same frozen validation set.

For example:

```text
Baseline validation
       =
Robust RGB validation
       =
RGB + FFT validation
```

---

### 6.3 Mandatory Metrics

Report:

* Accuracy
* AUROC
* False Positive Rate (FPR)

Also report robustness degradation where appropriate.

---

### 6.4 FPR Definition

For Verixa:

```text
Positive class = AI-GENERATED
Negative class = REAL
```

FPR represents the fraction of REAL images incorrectly flagged as AI-generated.

If using FPR at a target recall/operating point, explicitly record the threshold-selection procedure.

For example:

> FPR at 95% AI recall = the percentage of REAL images classified as AI at the threshold achieving 95% recall for AI-generated images.

---

### 6.5 Robustness Degradation

Where appropriate, report the change relative to clean validation performance:

```text
Δ metric = transformed metric − clean metric
```

This makes it easier to understand how much performance is lost after each transformation.

---

### 6.6 Full Robustness Matrix

The final robustness report must include:

```text
             Clean
               │
     ┌─────────┼─────────┐
     ↓         ↓         ↓
   JPEG      Blur      Resize
     ↓         ↓         ↓
   Noise     Color      Crop
```

with all specified severity levels.

Do not report only one aggregate robustness number.

---

### 6.7 Held-Out Benchmark

COCO val2017 + DALL·E Advanced must only be evaluated after the model-selection process is complete.

Do not repeatedly evaluate the held-out benchmark while tuning the model.

The held-out benchmark is for final demonstration/evaluation, not experimentation.

---

### 6.8 Error Analysis

Analyze representative:

* false positives
* false negatives

from validation and final evaluation where permitted.

Document possible causes such as:

* unusual real-image processing
* unseen generator characteristics
* compression
* blur
* loss of useful image signal
* dataset bias

Do not present speculative explanations as confirmed facts.

---

## 7. Error Handling & Operational Rules

### 7.1 Fail Loudly on Data Errors

If a required dataset file, manifest, or image is corrupted or missing:

**fail with an informative error.**

Do not silently continue with corrupted or missing data.

---

### 7.2 No Silent Sample Drops

If an image fails during ingestion:

* record its ID/path
* record the reason
* count the failure
* report the failure rate

If the error rate exceeds **1% of the target batch**, halt ingestion and investigate.

---

### 7.3 Report Actual Numbers

Use measured values whenever possible.

For example, report:

```text
Actual images = 8,000
Actual disk usage = 91.31 MB
```

rather than estimates when the filesystem can be measured directly.

The same rule applies to:

* dataset size
* model parameter count
* VRAM usage
* training time
* failed samples
* duplicate counts

---

### 7.4 Verify Agent/Subagent Findings

If an AI subagent reports a result about:

* files
* datasets
* code
* model configuration
* experiment results

verify the claim against the actual filesystem, repository, or command output before accepting it as fact.

---

## 8. Experiment Management Rules

### 8.1 Every Important Experiment Must Be Reproducible

Record:

* experiment name
* date/time where useful
* dataset/manifest
* seed
* model
* hyperparameters
* transformations
* checkpoint
* metrics
* hardware information

---

### 8.2 Do Not Overwrite Important Results

Do not overwrite previous experiment results without explicit approval.

Prefer experiment-specific output directories/files.

For example:

```text
reports/
├── baseline_clean_eval.json
├── robust_rgb_eval.json
└── fft_hybrid_eval.json
```

---

### 8.3 Decision Checkpoints

There are two major model-selection checkpoints.

#### Checkpoint 1

Compare:

```text
Clean RGB ConvNeXt
        VS
Transformation-aware RGB ConvNeXt
```

using the same frozen dataset and validation set.

#### Checkpoint 2

Compare:

```text
Robust RGB
     VS
Robust RGB + FFT
```

using the same frozen dataset and validation set.

The FFT model is kept only if the measured results justify its additional complexity.

---

### 8.4 Negative Results Must Be Preserved

If an experiment does not help, do not delete it from the project history.

A result such as:

> "FFT did not improve robustness under JPEG and blur"

is a legitimate engineering finding and may strengthen the final technical explanation.

---

## 9. Git Commit Conventions

Commit messages must follow this exact phase-based structure:

| Phase       | Designated Git Commit Message                                                             |
| :---------- | :---------------------------------------------------------------------------------------- |
| **Phase 1** | `feat(data): complete Phase 1 dataset ingestion, manifests, and deterministic split`      |
| **Phase 2** | `feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training and evaluation`        |
| **Phase 3** | `feat(training): complete Phase 3 transformation-aware training and Checkpoint 1`         |
| **Phase 4** | `feat(exp): complete Phase 4 FFT dual-branch experiment and Checkpoint 2`                 |
| **Phase 5** | `feat(eval): complete Phase 5 comprehensive robustness and held-out benchmark evaluation` |
| **Phase 6** | `feat(analysis): complete Phase 6 error analysis and failure mode breakdown`              |
| **Phase 7** | `feat(inference): complete Phase 7 standalone inference pipeline and CLI`                 |
| **Phase 8** | `docs(polish): complete Phase 8 documentation, report summaries, and submission polish`   |

Do not commit a phase until its completion criteria have been verified.

---

# 10. Core Principle

When in doubt, prioritize:

```text
Correctness
    ↓
Reproducibility
    ↓
Fair experimentation
    ↓
Robustness
    ↓
Simplicity
    ↓
Innovation
```

Do not sacrifice experimental correctness merely to make the project appear more sophisticated.

**The goal of Verixa is not to build the most complicated detector. The goal is to build, measure, and honestly demonstrate a detector that remains reliable when AI-generated images undergo realistic transformations.**
