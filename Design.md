# Visual Identity & Technical Design Specification — Verixa

> [!NOTE]
> Verixa is an applied machine learning and computer vision research project. This document defines the visual identity, documentation aesthetics, diagramming rules, chart styling, and presentation conventions across all repository outputs. It is a visual specification, not a directive to build web frontends or add UI dependencies.

---

## 1. Design Philosophy & Tone

The visual design language of Verixa communicates **forensic rigor, mathematical transparency, structural robustness, and technical credibility**.

### Core Attributes
- **Forensic & Methodical:** Clean, structured layouts reminiscent of scientific publications and specialized analytical tools.
- **Data-Dense & Legible:** High information density without visual clutter; emphasis on high-contrast tables and clear metric hierarchies.
- **Anti-Hype Aesthetic:** Strictly avoid generic "futuristic neon glow", glowing cyber brains, or marketing buzzword aesthetics. Focus on precise graphs, exact numbers, and empirical evidence.

---

## 2. Color Palette

The color system is optimized for accessibility, dark/light markdown rendering, and clear semantic differentiation in technical plots:

| Role | Color Name | Hex Code | Usage & Semantic Meaning |
| :--- | :--- | :--- | :--- |
| **Primary Brand** | Deep Slate Blue | `#1E293B` | Core structural headers, primary architecture nodes, dominant theme |
| **Authentic / Real** | Forest Jade | `#0D9488` | Class `0` (REAL / Authentic imagery), positive integrity indicators |
| **Synthetic / AI** | Amber Crimson | `#E11D48` | Class `1` (AI-GENERATED imagery), synthetic anomaly detection |
| **Accent / Focus** | Cobalt Indigo | `#3B82F6` | Primary branch highlights, active transformation curves, hyperlinks |
| **Neutral Dark** | Charcoal Carbon | `#0F172A` | Terminal text, code block backgrounds, primary typography |
| **Neutral Light** | Clean Surface | `#F8FAFC` | Plot background canvas, secondary table shading |
| **Border / Divider**| Cool Slate Grey | `#CBD5E1` | Table borders, diagram connectors, grid lines |
| **Warning / Alert** | Ochre Amber | `#D97706` | High-risk degradation, severe distortion alerts, decision gates |

---

## 3. Typography & Hierarchy

### 3.1 Typeface Families
- **Prose & Documentation:** System Sans-Serif stack: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`, `sans-serif`.
- **Code, Hashes & Metrics:** High-legibility monospaced stack: `JetBrains Mono`, `Fira Code`, `Cascadia Code`, `Consolas`, `monospace`.
- **Mathematical Formulations:** KaTeX / LaTeX typesetting for loss functions, Fourier transforms, and evaluation formulas.

### 3.2 Heading Hierarchy
- `# Title (H1)`: Used strictly for document root titles.
- `## Section (H2)`: Major functional modules and phase gates.
- `### Subsection (H3)`: Specific technical components, datasets, and experiment stages.
- `#### Component (H4)`: File-level operations and granular task items.

---

## 4. Documentation & Table Formatting Conventions

### 4.1 Markdown Table Style
Tables must be structured with explicit alignment, bold header text, and clear metric units:

```markdown
| Source Dataset | Split | Images | Real (0) | AI-Gen (1) | On-Disk Size | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CIFAKE** | Train | 3,000 | 1,500 | 1,500 | 17.95 MB | Complete |
| **SID_Set** | Train | 3,000 | 1,500 | 1,500 | 43.75 MB | Complete |
| **WildFake** | Train | 2,100 | 1,000 | 1,100 | ~18.00 MB | Streaming |
| **Total Pilot**| — | **8,100** | **4,000** | **4,100** | **~79.70 MB** | Ready |
```

### 4.2 GitHub Alert Usage
Use GitHub-style alert callouts with precise semantic intent:
- `> [!NOTE]`: Background context, implementation notes, or provenance details.
- `> [!TIP]`: Performance optimizations, caching strategies, or speedups.
- `> [!IMPORTANT]`: Binding project constraints, parameter limits, or protocol rules.
- `> [!WARNING]`: Performance degradation alerts, OOM risks, or experimental caveats.
- `> [!CAUTION]`: Actions that could cause data leakage, split contamination, or disk overflow.

---

## 5. Chart & Plotting Conventions (Matplotlib / Seaborn)

When generating evaluation figures in `reports/figures/`:
1. **Style Base:** `seaborn-v0_8-whitegrid` with crisp 1.0pt grid lines (`#E2E8F0`).
2. **Resolution:** Minimum 300 DPI for publication/presentation clarity (`dpi=300, bbox_inches='tight'`).
3. **Line Plots (Robustness Degradation):**
   - **X-axis:** Distortion severity level (e.g., JPEG Quality: Clean, 90, 70, 50, 30).
   - **Y-axis:** Metric value (Accuracy or AUROC $\in [0.0, 1.0]$).
   - **Solid Line with Circular Markers:** Robust model.
   - **Dashed Line with Square Markers:** Clean baseline model.
   - **Shaded Region:** Degradation delta ($\Delta_{\text{degradation}}$).
4. **ROC Curves:**
   - Display False Positive Rate on X-axis and True Positive Rate on Y-axis.
   - Include diagonal reference line (`linestyle='--', color='#94A3B8'`) representing random guessing.
   - Annotate operating point threshold ($p = 0.5$) with a distinct star marker.
5. **Confusion Matrices:**
   - Normalized by true class row totals (percentages).
   - Colormap: `Blues` or custom `Jade-to-Crimson` gradient.

---

## 6. Architecture & Flow Diagram Conventions

When creating ASCII or Mermaid diagrams in documentation:
- **Direction:** Top-to-bottom (`TB` or `TD`) for end-to-end data flows; left-to-right (`LR`) for neural network sub-blocks.
- **Node Geometry:**
  - `[...]` (Rectangles): Functional components, models, and scripts.
  - `(...)` (Rounded Rectangles): Data stores, manifests, and datasets.
  - `{...}` (Rhombus): Decision checkpoints and conditional branches.
- **Connectors:** Explicit text annotations on decision branches (e.g., `|Keep FFT|`, `|Reject FFT|`).

---

## 7. Standard Terminology & Nomenclature

To ensure consistent communication across all reports, code, and documentation:

| Preferred Term | Deprecated / Disallowed Terms | Context & Rationale |
| :--- | :--- | :--- |
| **AI-Generated (AIGC)** | "Fake", "Forged", "Deepfake" | Generic terms conflate tampering with generation. Use AI-Generated for synthetic media. |
| **Authentic / Real** | "Original", "Genuine", "Clean" | "Clean" refers to lack of noise/distortion; "Authentic/Real" refers to non-synthetic ground truth. |
| **Robustness Degradation ($\Delta$)** | "Accuracy drop", "Loss" | "Degradation" specifically quantifies the performance delta under distortion. |
| **Generator Architecture** | "Generator Model", "Engine" | In WildFake, metadata represents the architectural family (DDPM, LDM, etc.), not always exact model weights. |
| **Held-Out Benchmark** | "Test Set", "Eval Data" | Distinguishes the isolated final benchmark from the in-domain validation split. |

