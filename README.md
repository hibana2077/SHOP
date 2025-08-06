# SHOP: Standardized Higher-Order Moment Pooling for Ultra-Fine-Grained CNNs

This repository implements **SHOP (Standardized Higher-Order Moment Pooling)**, a lightweight plug-and-play module that enhances CNN backbones with higher-order statistical moments (3rd and 4th order) for ultra-fine-grained classification tasks.

## Overview

SHOP addresses the challenge of ultra-fine-grained classification where:
- **Inter-class differences are extremely small**
- **Intra-class variations are large**
- **Traditional 1st and 2nd order statistics are insufficient**

The method computes standardized higher-order central moments (skewness/kurtosis) along with optional low-rank covariance pooling, providing better discriminative features for fine-grained classification.

## Key Features

- 🔌 **Plug-and-play**: Works with any timm CNN backbone
- ⚡ **Lightweight**: Only O(C) memory overhead and O(CHW) computation
- 📊 **Higher-order statistics**: 3rd and 4th order standardized moments
- 🎯 **Targeted for ultra-FGVC**: Designed for Ultra-Fine-Grained Visual Classification
- 🛠️ **Easy integration**: Simple configuration-based training

## Architecture

```
CNN Backbone (timm) → SHOP Head → Classifier
                        ↓
    [GAP, 3rd moments, 4th moments, cross-channel moments, optional low-rank cov]
```

### SHOP Head Components:
1. **Per-channel standardized moments**: μ³ and μ⁴ for each feature channel
2. **Cross-channel moments**: Higher-order statistics across channels via random projection
3. **Optional low-rank covariance**: Lightweight 2nd-order pooling
4. **Signed square-root + L2 normalization**: For numerical stability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/SHOP.git
cd SHOP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python demo.py
```

## Datasets

The implementation supports the Ultra-FGVC benchmark datasets:
- **Cotton80**: 80 cotton varieties
- **Soybean**: Soybean classification
- **SoyAgeing R1-R6**: Soybean aging datasets

Datasets are automatically downloaded from HuggingFace Hub when needed.

## Quick Start

### 1. Basic Usage

```python
from src.models.shop import create_shop_model

# Create SHOP model with ResNet-50 backbone
model = create_shop_model(
    backbone_name='resnet50',
    num_classes=80,  # Number of classes in your dataset
    pretrained=True,
    proj_dim=32,
    use_low_rank_cov=True
)

# Forward pass
import torch
x = torch.randn(4, 3, 224, 224)  # Batch of images
outputs = model(x)  # (4, 80) logits
```

### 2. Using Predefined Configurations

```python
from src.models.shop import create_shop_model_from_config

# Use predefined configuration
model = create_shop_model_from_config(
    config_name='shop_convnext_tiny',
    num_classes=80
)
```

### 3. Training

```bash
# Train on Cotton80 dataset with ConvNeXt-Tiny
python train.py --config configs/cotton80_convnext_tiny.yaml

# Train on Cotton80 dataset with ResNet-50
python train.py --config configs/cotton80_resnet50.yaml
```

### 4. Evaluation

```bash
# Evaluate trained model
python evaluate.py \
    --config configs/cotton80_convnext_tiny.yaml \
    --checkpoint experiments/cotton80_convnext_tiny_shop/checkpoints/best_checkpoint.pth \
    --split test \
    --save_predictions
```

## Configuration

Training configurations are stored in YAML files. Example:

```yaml
experiment:
  name: "cotton80_convnext_tiny_shop"
  output_dir: "./experiments"

dataset:
  name: "cotton80"
  root: "./data"
  download: true

model:
  config_name: "shop_convnext_tiny"
  pretrained: true

training:
  batch_size: 32
  epochs: 200
  optimizer: "adamw"
  learning_rate: 0.001
  scheduler: "cosine"
```

## Supported Backbones

The implementation supports all timm CNN models. Predefined configurations include:

- `shop_resnet50` / `shop_resnet101`
- `shop_densenet201`
- `shop_convnext_tiny` / `shop_convnext_small`
- `shop_efficientnet_b3`
- `shop_regnetx_032`

## Model Variants

| Variant | Global Pooling | 2nd Order | 3rd Order | 4th Order |
|---------|----------------|-----------|-----------|-----------|
| **B0** (Baseline) | GAP | – | – | – |
| **B1** | fast-MPN-COV | ✓ | – | – |
| **S3/4** (SHOP) | GAP | – | ✓ | ✓ |
| **S2+3/4** (SHOP+) | GAP + low-rank Cov | ✓ (low-rank) | ✓ | ✓ |

## Experimental Results

Expected performance improvements on ultra-fine-grained datasets:
- **SoyLocal, Cotton80, SoyGlobal**: Significant improvements over baseline GAP and 2nd-order methods
- **SoyAgeing, SoyGene**: Competitive or improved performance
- **Computational overhead**: Minimal increase in FLOPs and parameters

## Project Structure

```
SHOP/
├── src/
│   ├── models/
│   │   └── shop.py          # SHOP model implementation
│   └── dataset/
│       └── ufgvc.py         # UFGVC dataset loader
├── configs/                 # Training configurations
│   ├── cotton80_convnext_tiny.yaml
│   ├── cotton80_resnet50.yaml
│   └── soybean_convnext_tiny.yaml
├── scripts/                 # Training scripts for HPC
├── docs/                    # Documentation
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── demo.py                  # Demo and testing
├── utils.py                 # Utility functions
└── requirements.txt         # Dependencies
```

## Mathematical Foundation

### Standardized Higher-Order Moments

For feature channel c, SHOP computes:

**3rd order (skewness):**
$$m^{(3)}_c = \frac{1}{N}\sum_{i=1}^N \left(\frac{x_{c,i}-\mu_c}{\sigma_c+\epsilon}\right)^3$$

**4th order (kurtosis):**
$$m^{(4)}_c = \frac{1}{N}\sum_{i=1}^N \left(\frac{x_{c,i}-\mu_c}{\sigma_c+\epsilon}\right)^4$$

### Properties:
- **Scale invariant**: Invariant to channel-wise scaling
- **Translation invariant**: Invariant to spatial permutations
- **Discriminative**: Can distinguish distributions with same mean/variance but different higher-order properties

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shop2024,
  title={SHOP: Standardized Higher-Order Moment Pooling for Ultra-Fine-Grained CNNs},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [timm](https://github.com/rwightman/pytorch-image-models) for the excellent model library
- Ultra-FGVC dataset authors for the benchmark datasets
- PyTorch team for the deep learning framework