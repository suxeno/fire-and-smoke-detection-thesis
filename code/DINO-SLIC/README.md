# DINO-SLIC <img src="figs/dinosaur.png" width="30">

**DINO-SLIC** is an experimental object detection model that replaces the traditional CNN backbone in [DINO](https://arxiv.org/abs/2203.03605) with a **handcrafted superpixel feature extractor** based on SLIC (Simple Linear Iterative Clustering). Instead of learning features through millions of convolution parameters, DINO-SLIC computes interpretable geometric, color, and texture descriptors from superpixels and feeds them as tokens directly into a standard transformer.

> **Key Idea:** Superpixels *are* the tokens. No grids, no convolutions — just handcrafted features + a lightweight transformer.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DINO-SLIC Pipeline                        │
│                                                                     │
│  Image ───► SLIC Segmentation ───► Feature Extraction ───► Tokens  │
│  (RGB)      (3 scales)              (113-dim per SP)       (256-d) │
│                                                                     │
│  Tokens + Centroids ───► MHSA Encoder ───► Cross-Attn Decoder     │
│                          (standard)        (with DN queries)       │
│                                                                     │
│  Decoder ───► Class Predictions + Bounding Boxes                   │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works (Step by Step)

| Step | What Happens | Code |
|------|-------------|------|
| **1. SLIC Segmentation** | Image is segmented into superpixels at 3 scales (`n_segments=[400, 200, 100]`), mimicking multi-scale CNN feature maps (C3/C4/C5). | `slic_backbone.py` |
| **2. Feature Extraction** | For each superpixel, a **113-dim feature vector** is computed covering color, shape, texture, spatial, and relational properties. | `slic_backbone.py` |
| **3. Projection** | A linear layer projects 113-dim → 256-dim to match the transformer's hidden dimension. | `slic_backbone.py` |
| **4. Token Output** | The backbone outputs `tokens [bs, N, 256]`, `centroids [bs, N, 2]`, `padding_mask [bs, N]`, and `level_counts`. | `slic_backbone.py` |
| **5. Positional Encoding** | Sinusoidal PE is computed from the normalized `(cx, cy)` centroid of each superpixel + per-level embeddings to distinguish scales. | `slic_transformer.py` |
| **6. Encoder** | Standard multi-head self-attention (MHSA) encoder processes all superpixel tokens together. No deformable attention needed. | `slic_transformer.py` |
| **7. Two-Stage Proposals** | Encoder output generates box proposals from centroids: each superpixel center → `(cx, cy, w, h)` with scale-dependent default sizes. Top-k proposals become decoder queries. | `slic_transformer.py` |
| **8. Decoder** | Cross-attention decoder: object queries attend to superpixel memory. Iterative box refinement updates reference points at each layer. | `slic_transformer.py` |
| **9. Detection Head** | Final class logits + bounding box regression. Supports denoising (DN) training from DINO. | `dino.py` |

---

## 🧠 Why Superpixels as Tokens?

Traditional DINO uses **deformable attention** on spatial grids (~8,400 grid cells across scales). This requires:
- Fixed spatial grids
- Bilinear sampling at learned offsets
- Custom CUDA kernels

DINO-SLIC instead uses **~700 superpixel tokens** with standard `nn.MultiheadAttention`:

| | Grid-based DINO | DINO-SLIC |
|---|---|---|
| **Token count** | ~8,400 (80² + 40² + 20²) | ~700 (400 + 200 + 100) |
| **Attention type** | Deformable (sparse, needs CUDA ops) | Standard MHSA (dense, pure PyTorch) |
| **Position encoding** | Grid coordinates | Superpixel centroids |
| **Backbone parameters** | ~23M (ResNet-50) | ~30K (linear projection only) |
| **Feature source** | Learned convolutions | Handcrafted descriptors |

---

## 📊 Feature Vector Breakdown (113-dim)

Each superpixel is described by 113 interpretable features:

| Category | Dims | What's Computed |
|:---------|:-----|:----------------|
| **Color** | 72 | Mean, Std, Skewness, Kurtosis per channel (RGB, Lab, HSV) + Entropy + Color Histograms (8 bins × 3 channels × 2 spaces) + Dominant Color |
| **Shape** | 12 | Area, Perimeter, Compactness, Aspect Ratio, Eccentricity, Solidity, Extent + Hu Moments (5) |
| **Texture** | 21 | LBP histogram (10 bins) + Gradient features (HOG-like, 9 orientation bins + magnitude stats) |
| **Spatial** | 4 | Normalized centroid `(cx, cy)` + Distance and angle from image center |
| **Relational** | 4 | Mean/max color contrast with neighbors + Boundary smoothness + Neighbor count ratio |

---

## � Project Structure

```
DINO-SLIC/
├── config/DINO/
│   └── DINO_4scale_slic.py        # SLIC-specific configuration
├── models/dino/
│   ├── slic_backbone.py           # SLIC segmentation + feature extraction
│   ├── slic_transformer.py        # Standard MHSA encoder/decoder
│   ├── dino.py                    # Main DINO model (SLIC branch integrated)
│   ├── backbone.py                # Backbone builder (routes to SLIC or CNN)
│   ├── dn_components.py           # Denoising training components
│   ├── deformable_transformer.py  # Original deformable transformer (CNN path)
│   └── utils.py                   # MLP, sinusoidal embeddings, etc.
├── datasets/                      # COCO-format dataset loaders
├── scripts/
│   └── DINO_train_slic.sh         # Training launch script
├── main.py                        # Training/evaluation entry point
├── engine.py                      # Train/eval loops
├── quick_test_slic.py             # Quick validation (3 smoke tests)
└── util/                          # Utilities (misc, box_ops, slconfig, etc.)
```

---

## 🔑 Key Source Files

### `slic_backbone.py` — The Backbone

The `SLICFeatureExtractor` replaces CNN backbones entirely. It:
1. Runs `skimage.segmentation.slic()` at 3 scales
2. Computes 113 handcrafted features per superpixel via `_compute_features()`
3. Projects to 256-dim via `FeatureProjection` (linear layer)
4. Pads variable-count superpixels across the batch
5. Returns a dict: `{tokens, centroids, padding_mask, level_counts}`

### `slic_transformer.py` — The Transformer

The `SLICTransformer` replaces `DeformableTransformer`. Key components:
- **`SuperpixelPositionEmbedding`** — Sinusoidal PE from `(cx, cy)` centroids
- **`SLICEncoder`** — Stack of standard MHSA layers
- **`SLICDecoder`** — Cross-attention layers with iterative box refinement
- **`gen_encoder_output_proposals_from_centroids()`** — Two-stage proposal generation where each superpixel centroid becomes a candidate detection

### `dino.py` — The Orchestrator

The `DINO` class detects SLIC via `isinstance(backbone, SLICFeatureExtractor)` and:
- **`__init__`**: Skips `input_proj` (no Conv2d needed), uses `SLICTransformer`
- **`forward`**: Calls backbone for token dict → passes directly to transformer
- **`build_dino`**: Routes to `SLICTransformer` when `args.backbone == 'slic'`

All denoising training, loss computation, and auxiliary outputs remain identical to standard DINO.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-image scipy torch torchvision
```

> **Note:** SLIC mode does **not** require the custom CUDA deformable attention ops (`models/dino/ops`). Those are only needed for the standard CNN backbone path.

### Quick Validation

Run 3 smoke tests (backbone → transformer → full model):

```bash
python3 quick_test_slic.py
```

Expected output:
```
TEST 1: Backbone Token Output       ✓
TEST 2: SLICTransformer Forward      ✓
TEST 3: Full DINO-SLIC Model (E2E)   ✓
ALL TESTS PASSED ✓
```

### Training

Train on FASDD (or any COCO-format dataset):

```bash
bash scripts/DINO_train_slic.sh
```

Or manually:
```bash
python3 main.py \
  --output_dir outputs/DINO-SLIC/run1 \
  -c config/DINO/DINO_4scale_slic.py \
  --coco_path /path/to/your/dataset \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False
```

### Evaluation

```bash
python3 main.py \
  --output_dir outputs/DINO-SLIC/run1 \
  -c config/DINO/DINO_4scale_slic.py \
  --coco_path /path/to/your/dataset \
  --eval --resume outputs/DINO-SLIC/run1/checkpoint.pth
```

---

## ⚙️ Configuration

Key parameters in `config/DINO/DINO_4scale_slic.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `'slic'` | Selects the SLIC backbone |
| `slic_n_segments` | `[400, 200, 100]` | Superpixels per scale (fine → coarse) |
| `slic_compactness` | `10.0` | SLIC compactness (higher = more regular shapes) |
| `slic_sigma` | `1.0` | Gaussian smoothing before segmentation |
| `num_feature_levels` | `3` | Matches the 3 SLIC scales |
| `hidden_dim` | `256` | Transformer hidden dimension |
| `enc_layers` | `6` | Number of encoder layers |
| `dec_layers` | `6` | Number of decoder layers |
| `num_queries` | `900` | Max detections per image |
| `two_stage_type` | `'standard'` | Two-stage proposal generation |

---

## 📚 References & Credits

This project is built upon the official DINO implementation:

> **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection**
> Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum
> [arXiv:2203.03605](https://arxiv.org/abs/2203.03605)

Original DINO is licensed under the Apache 2.0 license.
