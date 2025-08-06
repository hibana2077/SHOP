## Document 2 – Planned Experiments & Data to Collect

> **Filename suggestion:** `shop_experimental_plan.md`

### 1. Data sets

* **Ultra-FGVC (UFG) benchmark – 5 subsets**

  * SoyAgeing · SoyGene · SoyLocal · SoyGlobal · Cotton80&#x20;
* Official train/val splits and Top-1 evaluation protocol.

### 2. CNN Backbones (all from timm)

ResNet-50/101, DenseNet-201, RegNet‐Y8G, ConvNeXt-T/S, EfficientNet-B3, RepVGG-B1, etc.&#x20;

### 3. Model Variants to Run

| ID         | Global-pooling head | Second-order | Third | Fourth |
| ---------- | ------------------- | ------------ | ----- | ------ |
| **B0**     | GAP (baseline)      | –            | –     | –      |
| **B1**     | fast-MPN-COV        | ✓            | –     | –      |
| **S3/4**   | SHOP (ours)         | –            | ✓     | ✓      |
| **S2+3/4** | SHOP + low-rank Cov | ✓ (low-rank) | ✓     | ✓      |

(Baseline list taken from proposal )

### 4. Ablation Studies

1. Remove σ-normalisation
2. Only third-order (m³)
3. Only fourth-order (m⁴)
4. Projection rank r ∈ {16, 32, 64}
5. Input resolution 224 vs 384 pixels&#x20;

### 5. Metrics to Collect (per run)

| Category               | Metric                                                          | Notes              |
| ---------------------- | --------------------------------------------------------------- | ------------------ |
| **Accuracy**           | Top-1, Top-5                                                    | val split          |
| **Learning dynamics**  | train/val loss curves, epoch convergence                        |                    |
| **Efficiency**         | Params, FLOPs, GPU memory, images/s                             | compare B0 vs S3/4 |
| **Robustness**         | Few-shot accuracy (1-shot, 5-shot), class-imbalance sensitivity | optional           |
| **Stat. significance** | 3-run mean ± σ, paired t-test vs baseline                       |                    |

### 6. Logging & Artifacts

* **Checkpoints**: best Top-1 and last epoch per backbone.
* **TensorBoard / Weights & Biases** runs with above metrics.
* **CSV summary tables** for paper (rows = backbones, cols = variants × datasets).

### 7. Expected Outcomes

* SHOP should outperform GAP and pure 2-nd-order heads on small-sample or ultra-fine subsets (SoyLocal, Cotton80, SoyGlobal) while matching performance on SoyAgeing & SoyGene, validating the high-order moment hypothesis .

---
