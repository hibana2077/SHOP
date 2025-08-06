Below are **two self-contained English documents** based on the attached SHOP proposal.
Feel free to copy them into separate `.md` files or any format you prefer.

---

## Document 1 – Training + Validation Flowchart

> **Filename suggestion:** `shop_training_validation_flowchart.md`

### Overview

The diagram shows the end-to-end loop used for every Ultra-FGVC subset and CNN backbone.
Hyper-parameters (optimizer, learning-rate schedule, augmentations, etc.) stay exactly the same as the original timm recipe—only the lightweight **SHOPHead** is inserted before the classifier .

```mermaid
graph TD
    A(Start) --> B[Load dataset split<br/>(SoyAgeing, SoyGene,<br/>SoyLocal, SoyGlobal, Cotton80)]
    B --> C[Data Loader<br/>(augment + batch)]
    C --> D[Create timm backbone<br/>(ResNet-50/101, DenseNet,<br/>ConvNeXt, EfficientNet, RepVGG…)]
    D --> E[Attach SHOPHead(C, r = 32)]
    E --> F[Forward pass<br/>compute μ, σ², m³, m⁴<br/>+ optional low-rank Cov]
    F --> G[Cross-entropy loss]
    G --> H[Optimizer step<br/>(SGD or AdamW)]
    H --> I[Log metrics & save checkpoint]
    I --> J{End of epoch?}
    J -- yes --> K[Run validation split<br/>Top-1 & loss]
    K --> L{Early-stop/best?}
    L -- continue --> C
    J -- no --> C
```

**Key notes**

* **SHOPHead** adds only O(C) memory and O(CHW) time per batch, so throughput stays near baseline .
* Validation tracks **Top-1 accuracy** (primary WACV metric) and loss on the official split .
* Early-stopping or learning-rate decay is triggered by validation loss plateau.

---
