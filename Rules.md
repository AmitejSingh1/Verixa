# Engineering Rules & Agent Guidelines — Verixa

> [!IMPORTANT]
> This is the foundational rulebook for all human developers and AI coding agents operating in the Verixa repository (`C:\Verixa`). Every instruction in this document is strict and binding.

---

## 1. General Workflow & Development Rules

1. **Strict Phase Sequencing:** Follow the eight development phases defined in `Phases.md` sequentially. Do **not** skip phases, jump ahead, or implement components from later phases out of order.
2. **Atomic Phase Commits:** Never combine multiple phases into a single git commit. Each phase must be committed individually with its exact designated commit message once its completion criteria are met.
3. **Repository Runnable at All Times:** Every commit and phase transition must leave the repository fully runnable. All existing unit tests and linter checks must pass cleanly.
4. **Test Before Progressing:** Run the full test suite (`pytest -v`) and linter (`ruff check .`) before marking any phase as complete.
5. **No Invented Requirements:** Do not add unapproved features, datasets, external services, or complex abstractions that have not been explicitly agreed upon.
6. **No Silent Changes:** Never change previously agreed configurations, seeds, thresholds, or architectural decisions without explicit user confirmation.

---

## 2. Dataset & Storage Rules

1. **No Bulk Source Downloads:** Never attempt to download entire multi-gigabyte source archives (e.g., the full 124 GB SID_Set or 3M-row WildFake). Always use streaming or controlled subsampling.
2. **Enforce Local Storage Limits:**
   - Pilot dataset storage must remain $< 500\text{ MB}$.
   - If scaled to 30K–50K images in later phases, total storage must remain $< 2.5\text{ GB}$.
3. **Standardize Immediately on Ingestion:** Never retain raw, high-resolution source images locally. All ingested images must be immediately resized to $224 \times 224$ and saved as compressed JPEG ($Q=90$).
4. **Preserve Dataset Provenance:** Every row in every manifest must record the exact `source_dataset` and `original_id` so any image can be traced back to its origin.
5. **No Invented Generator Metadata:**
   - If a source dataset provides generator/architecture metadata (e.g., WildFake's `Architecture` field), record it faithfully as generator architecture/family metadata.
   - If a dataset does not provide generator metadata (e.g., SID_Set), record it as empty (`None`) or generic category (`full_synthetic`). Never fabricate specific generator model names.
6. **Strict Isolation of Held-Out Benchmark:** The held-out benchmark (COCO val2017 authentic + DALL-E Advanced synthetic) must **never** be accessed, sampled, or used during training, validation tuning, threshold selection, or intermediate development. It is strictly reserved for Phase 5 final evaluation.
7. **Mandatory Leakage & Deduplication Checks:** Run exact SHA-256 grouping and perceptual near-duplicate scanning on every manifest before training. Never split exact duplicate images across train and validation splits.

---

## 3. Machine Learning & Modeling Rules

1. **ConvNeXt-Tiny is the Primary Model:** The primary, fallback-ready architecture is pretrained ConvNeXt-Tiny fine-tuned on RGB imagery.
2. **FFT is Strictly Experimental:** The dual-branch RGB + FFT model is an experimental hypothesis to be evaluated in Phase 4. It must **not** be assumed to be the final model. It will only be adopted if it provides measurable, statistically meaningful robustness improvement over the RGB model at Decision Checkpoint #2.
3. **No Training from Scratch:** Always initialize vision backbones with official pretrained weights (e.g., ImageNet-1K). Training from random initialization on small datasets is prohibited.
4. **Strict Parameter Limit ($< 2\text{B}$):** Total model parameter count must remain strictly below the competition's 2 Billion parameter ceiling. ConvNeXt-Tiny is ~28.6M parameters.
5. **No Unjustified Ensembles:** Do not build large multi-model ensembles. A single robust model is the primary goal.
6. **Reproducibility & Fixed Seeds:** Fix random seed `1337` everywhere: Python `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `DataLoader` worker initialization functions.
7. **Hardware Headroom & VRAM Management:**
   - All training must execute on the local NVIDIA RTX 4060 Laptop GPU (8 GB VRAM).
   - Use mixed precision (`torch.amp.autocast`).
   - Use batch sizes between 16 and 32.
   - Freeze early backbone stages initially; unfreeze deeper layers progressively only if needed.
   - If an experimental configuration risks exceeding 7.2 GB VRAM or causes an OOM exception, stop and report the issue rather than silently altering core parameters.

---

## 4. Evaluation & Reporting Rules

1. **Zero Cherry-Picking:** All reported metrics must reflect uncurated, objective evaluations across full evaluation sets. Never discard poorly performing classes or severities from reports.
2. **Fixed Validation Set:** Always evaluate candidate models and baseline checkpoints on the exact same fixed validation set (`val` split of the merged manifest).
3. **Evaluate All 6 Transformations at All Severities:** Robustness reports must include evaluations across all competition-mandated transformations:
   - JPEG Compression: $Q \in \{90, 70, 50, 30\}$
   - Gaussian Blur: $\sigma \in \{0.5, 1.0, 2.0\}$
   - Resize: $0.5\times, 0.25\times$
   - Gaussian Noise: $\sigma \in \{0.02, 0.05, 0.10\}$
   - Color Jitter: $\pm 20\%$
   - Center Crop: $80\%$
4. **Mandatory Triad of Metrics:** Always compute and report:
   - **Accuracy**
   - **AUROC**
   - **False Positive Rate (FPR)** at operational decision thresholds
   - **Robustness Degradation ($\Delta$):** Drop in metric values relative to clean validation performance.
5. **No Repeated Over-Tuning on Benchmark:** Do not iteratively tweak hyperparameters against the held-out benchmark. Run the held-out benchmark only once at the conclusion of evaluation.

---

## 5. Error Handling & Operational Rules

1. **Fail Loudly on Data Errors:** If a dataset manifest is corrupted, missing files, or mislabeled, fail immediately with an informative exception. Never silently swallow errors.
2. **No Silent Sample Drops:** If an image fails to load or download during ingestion, log the specific file ID/URL and reason. If the error rate exceeds 1% of the target batch, halt the process immediately.
3. **Report Actual Numbers:** Never report mathematical estimations or projections when actual values can be measured directly (e.g., report actual on-disk folder sizes and actual dataset row counts).
4. **Subagent Execution Safety:** When subagents are spawned for research or inspection, their findings must be verified against actual filesystem and code state before accepting their conclusions.

---

## 6. Git Commit Conventions

Commit messages must strictly follow the standard phase-based structure:

| Phase | Designated Git Commit Message |
| :--- | :--- |
| **Phase 1** | `feat(data): complete Phase 1 dataset ingestion, manifests, and deterministic split` |
| **Phase 2** | `feat(model): complete Phase 2 RGB ConvNeXt-Tiny baseline training and evaluation` |
| **Phase 3** | `feat(training): complete Phase 3 transformation-aware training and Checkpoint 1` |
| **Phase 4** | `feat(exp): complete Phase 4 FFT dual-branch experiment and Checkpoint 2` |
| **Phase 5** | `feat(eval): complete Phase 5 comprehensive robustness and held-out benchmark evaluation` |
| **Phase 6** | `feat(analysis): complete Phase 6 error analysis and failure mode breakdown` |
| **Phase 7** | `feat(inference): complete Phase 7 standalone inference pipeline and CLI` |
| **Phase 8** | `docs(polish): complete Phase 8 documentation, report summaries, and submission polish` |

