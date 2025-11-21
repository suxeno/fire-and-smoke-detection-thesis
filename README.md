# DETR-SLIC: Superpixel-Enhanced DETR for Fire and Smoke Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **DETR-SLIC**: An efficient object detection model that fuses **DETR (DEtection TRansformer)** with **SLIC superpixels** for improved time efficiency in fire and smoke detection tasks.

## 🔥 Overview

This project implements DETR-SLIC, a novel architecture that enhances the standard DETR model by incorporating SLIC (Simple Linear Iterative Clustering) superpixels to reduce computational complexity while maintaining or improving detection accuracy.

### Key Features

- **🚀 Faster Training**: Reduces encoder tokens from ~2000 pixels to ~100 superpixels (20x reduction)
- **📊 Comparable Accuracy**: Maintains detection performance through superpixel-aware feature pooling
- **⚙️ Easy Configuration**: YAML-based configs for simplified experiment management
- **🎯 Fire & Smoke Detection**: Specialized for fire and smoke detection with 2 classes
- **📈 Comprehensive Logging**: Built-in training logs, checkpointing, and COCO evaluation metrics
- **🎨 Visualization Tools**: Automatic prediction visualization with ground truth comparison

---

## 📁 Project Structure

```
.
├── main.py                     # Training script (YAML config-based)
├── test.py                     # Testing/evaluation script with visualization
├── engine.py                   # Training and evaluation functions
├── requirements.txt            # Python dependencies
├── configs/
│   ├── detr_slic.yaml         # DETR-SLIC configuration
│   └── detr_baseline.yaml      # Baseline DETR configuration
├── models/
│   ├── detr_slic.py           # DETR-SLIC model implementation
│   ├── detr.py                # Baseline DETR model
│   ├── backbone.py            # ResNet backbone
│   ├── transformer.py         # Transformer encoder/decoder
│   └── superpixel/            # SLIC superpixel modules
│       ├── slic.py            # SLIC superpixel generation
│       ├── feature_pooling.py # Superpixel feature pooling
│       └── superpixel_encoder.py # Superpixel transformer encoder
├── util/
│   ├── data_loader.py         # COCO format dataset loader
│   ├── data_transform.py      # Image transformations
│   ├── config.py              # YAML config utilities
│   ├── misc.py                # Training utilities
│   └── plot_utils.py          # Visualization utilities
├── datasets/                   # Full dataset (122K images)
│   ├── images/                # Hierarchical image structure
│   │   ├── CV/                # Camera View images
│   │   ├── UAV/               # Drone images
│   │   └── RS/                # Remote Sensing images
│   └── annotations/
│       └── COCO/
│           └── Annotations/
│               ├── train.json
│               ├── val.json
│               └── test.json
├── datasets_sample/            # Sample dataset (2K images)
└── outputs/                    # Training outputs
    ├── detr_slic/             # DETR-SLIC results
    └── detr_baseline/         # Baseline DETR results
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd TA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install pycocotools for COCO metrics
pip install pycocotools
```

### 2. Dataset Preparation

The dataset follows a hierarchical structure:
```
images/
  {category}/      # CV, UAV, or RS
    {label}/       # fire, smoke, bothFireAndSmoke, neitherFireNorSmoke
      image.jpg
```

Annotations are in COCO format with 2 classes:
- **Class 0**: Fire
- **Class 1**: Smoke

### 3. Train DETR-SLIC (Sample Dataset)

```bash
# Quick training on 2K sample dataset
python main.py --config configs/detr_slic.yaml
```

### 4. Train on Full Dataset

```bash
# Edit configs/detr_slic.yaml:
# use_sample: false
# data_path: 'datasets'

python main.py --config configs/detr_slic.yaml
```

### 5. Evaluate Model

```bash
# Basic evaluation
python test.py \
  --config configs/detr_slic.yaml \
  --checkpoint outputs/detr_slic/checkpoint_best.pth

# With visualizations
python test.py \
  --config configs/detr_slic.yaml \
  --checkpoint outputs/detr_slic/checkpoint_best.pth \
  --visualize \
  --num-visualize 50 \
  --threshold 0.5
```

---

## 🎯 Model Architecture

### DETR-SLIC Pipeline

```
Input Image (3, H, W)
    ↓
ResNet50 Backbone
    ↓
Feature Map (2048, H/32, W/32)
    ↓
[SLIC Superpixel Generation] ← Novel Component
    ↓
Superpixel Maps (~100 segments)
    ↓
[Feature Pooling within Superpixels] ← Novel Component
    ↓
Superpixel Features (100, 256)
    ↓
[Superpixel Transformer Encoder] ← Novel Component
    ↓
Memory (100, 256)
    ↓
Standard Transformer Decoder
    ↓
Detection Heads (class + bbox)
    ↓
Predictions (100 queries)
```

### Key Innovations

1. **SLIC Superpixel Generation**: Groups pixels into ~100 perceptually meaningful regions
2. **Feature Pooling**: Mean-pools backbone features within each superpixel
3. **Superpixel Encoder**: Transformer encoder operating on superpixel tokens instead of pixels
4. **Spatial Encoding**: 4D spatial features (center_x, center_y, width, height) for each superpixel

---

## ⚙️ Configuration

All hyperparameters are managed through YAML config files:

```yaml
# Key parameters in configs/detr_slic.yaml

# Model
use_slic: true              # Enable SLIC superpixels
n_superpixels: 100          # Number of superpixels
slic_compactness: 10.0      # SLIC compactness parameter
pooling_method: 'mean'      # Feature pooling method

# Training
epochs: 300
batch_size: 2
lr: 1.0e-4
lr_backbone: 1.0e-5
lr_drop: 200

# Dataset
use_sample: true            # Use sample or full dataset
data_path: 'datasets_sample'
num_classes: 2

# Checkpointing
eval_every: 5               # Validate every N epochs
save_every: 10              # Save checkpoint every N epochs
```

---

## 📊 Training Monitoring

### Training Output

```
================================================================================
Configuration for detr_slic
================================================================================
Model: detr_slic
Use SLIC: True
Dataset: datasets_sample
Num classes: 2
Batch size: 2
Epochs: 300
Output: outputs/detr_slic
================================================================================

Building model...

============================================================
detr_slic Parameter Summary
============================================================
Total parameters:        41,234,567
Trainable parameters:    41,234,567
Non-trainable parameters: 0
============================================================

Loading datasets...
Training samples: 1000
Validation samples: 667

Epoch: [0]  [50/500]  lr: 0.000100  loss: 35.23  loss_ce: 0.65  loss_bbox: 3.21  loss_giou: 1.88  class_error: 95.43

Epoch 0/300 Summary:
  Train loss: 32.1234
  Val loss: 28.3456
  Val AP: 0.1234
  Val AP50: 0.2345
  Time: 123.5s
  ✓ Saved best checkpoint (val_loss: 28.3456)
```

### Output Files

```
outputs/detr_slic/
├── config.yaml              # Training configuration
├── checkpoint_latest.pth    # Latest checkpoint
├── checkpoint_best.pth      # Best checkpoint (lowest val loss)
├── checkpoint_epoch_0010.pth # Periodic checkpoints
├── log.txt                  # JSON training log
└── test_results.json        # Final test metrics
```

---

## 📈 Results

### Performance Comparison (Sample Dataset)

| Model | AP | AP50 | AP75 | Training Time/Epoch | Params |
|-------|-----|------|------|---------------------|--------|
| DETR Baseline | 0.xxx | 0.xxx | 0.xxx | XXX sec | 41M |
| **DETR-SLIC** | **0.xxx** | **0.xxx** | **0.xxx** | **XXX sec** | **41M** |

*Note: Update with your actual results*

### COCO Evaluation Metrics

When `pycocotools` is installed, the following metrics are computed:
- **AP**: Average Precision @ IoU=0.50:0.95
- **AP50**: AP @ IoU=0.50
- **AP75**: AP @ IoU=0.75
- **AP_small/medium/large**: AP for different object sizes

---

## 🔬 Experiments

### Compare DETR-SLIC vs Baseline

```bash
# Train DETR-SLIC
python main.py --config configs/detr_slic.yaml

# Train baseline DETR
python main.py --config configs/detr_baseline.yaml

# Compare results
python test.py --config configs/detr_slic.yaml --checkpoint outputs/detr_slic/checkpoint_best.pth
python test.py --config configs/detr_baseline.yaml --checkpoint outputs/detr_baseline/checkpoint_best.pth
```

### Ablation Studies

Create custom configs to test different settings:

```yaml
# configs/detr_slic_50sp.yaml
n_superpixels: 50

# configs/detr_slic_200sp.yaml
n_superpixels: 200

# configs/detr_slic_max_pooling.yaml
pooling_method: 'max'
```

---

## 🎨 Visualization

Test script generates side-by-side visualizations:

```bash
python test.py \
  --config configs/detr_slic.yaml \
  --checkpoint outputs/detr_slic/checkpoint_best.pth \
  --visualize \
  --num-visualize 50
```

Output: `outputs/detr_slic/test_results/visualizations/`
- Left: Ground truth (red=fire, yellow=smoke)
- Right: Predictions with confidence scores

---

## 🛠️ Advanced Usage

### Resume Training

```yaml
# Edit config file
resume: 'outputs/detr_slic/checkpoint_latest.pth'
start_epoch: 50
```

### Evaluation Only

```yaml
eval: true
resume: 'outputs/detr_slic/checkpoint_best.pth'
```

### Multi-GPU Training

```yaml
distributed: true
world_size: 2  # Number of GPUs
```

### CPU Training (for testing)

```yaml
device: 'cpu'
batch_size: 1
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={DETR-SLIC: Superpixel-Enhanced DETR for Efficient Fire and Smoke Detection},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [DETR (DEtection TRansformer)](https://github.com/facebookresearch/detr) by Facebook AI Research
- [SLIC Superpixels](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf)
- Fire and Smoke Detection Dataset contributors

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

## 🔖 Project Status

- ✅ DETR-SLIC model implementation
- ✅ Training pipeline with YAML configs
- ✅ COCO evaluation metrics
- ✅ Visualization tools
- ✅ Sample dataset (2K images)
- ✅ Full dataset support (122K images)
- 🔄 Ablation studies (in progress)
- 🔄 Paper submission (planned)

---

**Built with ❤️ for efficient fire and smoke detection** 🔥💨
