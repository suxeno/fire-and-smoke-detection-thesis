# DINO-SLIC <img src="figs/dinosaur.png" width="30">

**DINO-SLIC** is an experimental object detection model that replaces the traditional CNN backbone in [DINO](https://arxiv.org/abs/2203.03605) with a **GPU-accelerated handcrafted superpixel feature extractor** based on SLIC (Simple Linear Iterative Clustering). Instead of learning features through millions of convolution parameters, DINO-SLIC computes interpretable geometric, color, and texture descriptors from superpixels entirely on the GPU and feeds them as tokens directly into a standard transformer.

> **Key Idea:** Superpixels *are* the tokens. No grids, no convolutions, no CUDA deformable attention — just handcrafted GPU features (`scatter_add`) + standard MHSA transformer.

---

## 📐 Architecture Overview

```mermaid
graph TD
    A[Image RGB GPU] --> B[Pre-computed SLIC Maps]
    B -->|3 scales e.g. 400, 200, 100| C
    A --> C
  C[GPU Feature Extraction<br/>scatter_add] -->|136-dim per SP (118 appearance + 18 geometry)| D[Linear Projection]
    D -->|256-dim| E[Token Sequences + Centroids]
    E -->|Tokens + Paddings| F[Standard MHSA Encoder]
    F -->|Encoder Memory| G[Two-stage Box Proposals<br/>from Centroids]
    F --> H[Standard Cross-Attn Decoder]
    G -->|Top-K Queries| H
    H -->|Iterative Box Refinement| I[Class & Bbox Predictions]
```

### How It Works (Step by Step)

| Step | What Happens | Source |
|------|-------------|------|
| **1. SLIC Segmentation Maps** | Instead of running SLIC on CPU dynamically, **pre-computed SLIC `.npz` maps** are loaded from the dataset for 3 scales (e.g., `400, 200, 100`), mimicking CNN multiscale features. (CPU fallback exists). | Dataset / `slic_backbone.py` |
| **2. GPU Feature Extraction** | For each superpixel map, raw handcrafted features are extracted *entirely on the GPU* using `torch.scatter_add_`: **118D appearance + 18D geometry = 136D raw**. This eliminates costly Python loops and CPU-GPU memory transfers. | `slic_backbone.py` |
| **3. Linear Projection** | Appearance (**118D**) and geometry (**18D**) are concatenated directly into a single **136D** handcrafted token, then `nn.Linear` projects **136D -> 256D** embedding tokens. | `slic_backbone.py` |
| **4. Token Output** | The backbone aggregates all tokens across scales and outputs `tokens [bs, N_total, 256]`, continuous `centroids [bs, N_total, 2]`, `padding_mask` (to mask missing superpixels up to `max_superpixels_per_level`), and `level_counts`. | `slic_backbone.py` |
| **5. Positional Encoding** | Soft sinusoidal 2D positional encoding is generated from the normalized `(cx, cy)` topological centroids of the superpixels along with learned per-level embeddings representing the scales. | `slic_transformer.py` |
| **6. MHSA Encoder** | A purely Standard `nn.MultiheadAttention` encoder handles self-attention between the 1D sequence of non-grid tokens. | `slic_transformer.py` |
| **7. Two-Stage Proposals** | Encoder output serves to generate dense box proposals: every superpixel continuous centroid `(cx, cy)` gets a scale-dependent default `(w, h)` assigned. The top-K scoring proposals become query references for the Decoder. | `slic_transformer.py` |
| **8. Standard Cross-Attn Decoder** | Standard `nn.MultiheadAttention` is used for **both** Self-Attention among object queries and Cross-Attention from queries to superpixel memory. Supports DINO's iterative box refinement updating references. The output sequences are correctly transposed to `batch-first` `[num_layers, bs, nq, d_model]` for matcher stability. | `slic_transformer.py` |
| **9. Detection Head & CDN** | Final target projection to logits and bounding boxes. Smoothly integrates with DINO's original Contrastive DeNoising (CDN) training framework. | `dino.py` |

---

## 🧠 Why Superpixels as Tokens?

Standard DINO utilizes **deformable attention** applied on rigid visual spatial grids (~8,400 query grid cells natively across 3+ scales). This approach mandates:
- Fixed 2D spatial mappings
- Bilinear interpolation sampling at learned offsets
- Specialized compiled custom CUDA kernels (`MultiScaleDeformableAttention`)

DINO-SLIC inherently eliminates all the above, functioning with pure standard PyTorch components:

| Feature Dimension | Grid-based DINO | DINO-SLIC |
|---|---|---|
| **Token count** | ~8,400 cells (80² + 40² + 20²) | ~700 tokens (400 + 200 + 100) |
| **Attention type** | Deformable (Sparse sampling, custom ops) | **Standard MHSA** (Dense interaction, pure PyTorch) |
| **Position encoding** | Rigid 2D grid integer coordinates | Continuous superpixel `(cx, cy)` centroids |
| **Backbone parameters** | ~23-40M (ResNet/Swin) | **~35K** (Linear layer projection only!) |
| **Memory Extraction** | Learned hierarchical convolution | Handcrafted parallel GPU statistics (`scatter_add`) |

---

## 📊 Feature Vector Breakdown (Raw 136-dim)

Each superpixel region first maps to a detailed **raw 136-dim** handcrafted vector computed analytically on the GPU.
That raw 136D vector is fed directly to the projection layer (`136D -> 256D`).

| Category | Dims | What's Computed |
|:---------|:-----|:----------------|
| **Color** | **81** | **Moments:** Mean, Std, Range (approx 2σ), Skewness, Kurtosis per channel across 3 color spaces (**RGB, Lab, HSV** = 45D)<br/>**Distributions:** Channel Entropy (9D), RGB Histograms (8 bins × 3 channels = 24D), Dominant Color (3D) |
| **Texture** | **35** | **Edges/Gradients:** Sobel mean/std (4D), Gradient Magnitude mean/std (2D), HOG histogram (9 bins)<br/>**Patterns:** GLCM approximation (contrast, energy, homogeneity, entropy = 4D), LBP histogram (16 bins) |
| **Shape** | **12** | **Geometric:** Area, Perimeter (boundary counts), Compactness, Eccentricity<br/>**Moments:** Var Y, Var X, Cov YX, Hu Moments (log-transformed 5 moments) |
| **Spatial** | **4** | **Location:** Normalized continuous centroid `(cx, cy)`<br/>**Relative:** Distance from center, Angle from image center |
| **Relational** | **4** | **Context:** Mean absolute color contrast with 4-way neighbors, Mean texture diff with neighbors, Boundary strength, Graph neighbor degree count |

---

## 📁 Project Structure

```
DINO-SLIC/
├── config/DINO/
│   └── DINO_4scale_slic.py        # DINO configuration leveraging 'slic' backbone
├── models/dino/
│   ├── slic_backbone.py           # Multi-scale GPU extractor (raw 136D -> 256D)
│   ├── slic_transformer.py        # Standard MHSA Encoder/Decoder & Superpixel Positional Embeds
│   ├── dino.py                    # Object detector root (branches CNN vs SLIC logic)
│   ├── backbone.py                # Wrapper loader
│   ├── matcher.py                 # Hungarian matching algorithms
│   └── dn_components.py           # Contrastive Denoising (CDN) module
├── datasets/                      # COCO-format APIs (Loads pre-computed .npz SLIC maps)
├── scripts/
│   └── DINO_train_slic.sh         # Launch scripts for batch training
├── main.py                        # Training & evaluation pipeline
├── engine.py                      # Train epochs and logging
└── quick_test_slic.py             # Sandbox script to validate backbone/transformer IO
```

---

## 🔑 Key Components Explained

### `slic_backbone.py` — The Pure GPU Extractor
Replaces standard CNNs (`ResNet50`) with `SuperpixelFeatureExtractorGPU`.
1. Translates pre-computed `slic_maps` to flattened mappings.
2. Extracts **raw handcrafted features** (`118D appearance + 18D geometry = 136D`) using GPU `torch.scatter_add_` directly against input RGB tensors.
3. Applies a `FeatureProjection` (`nn.Linear(136 -> 256)`) encased in `LayerNorms`.
4. Pads token sequences (up to `max_superpixels_per_level`) allowing batching of differently segmented variable-size image data.
5. Emits `{'tokens', 'centroids', 'padding_mask', 'level_counts'}`.

### `slic_transformer.py` — The Standardized Transformer
Eliminates grid-reliant MultiScale Deformable Attentions with pure PyTorch equivalents:
- **`SuperpixelPositionEmbedding`**: Infers sin/cos traits mathematically from non-grid `[0,1]` relative `(cx, cy)` coordinates rather than arbitrary box cells.
- **`SLICEncoder` & `SLICDecoder`**: Comprised entirely of generic O(n²) `nn.MultiheadAttention` mapping tokens directly against memory items.
- **`gen_encoder_output_proposals_from_centroids`**: Creates Region Proposals logically anchored on centroids instead of uniformly spreading boxes over space grids.
- Outputs rigorously conform to `batch-first` rules expected by DINO criterion checks.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-image scipy torch torchvision numpy
```

> **Note:** Compiling DINO's custom `MultiScaleDeformableAttention` CUDA extensions inside `models/dino/ops` is **NOT REQUIRED**. DINO-SLIC bypasses that code branch entirely.

### Quick Validation

To verify that the GPU scatter extraction and the Transformer IO works correctly on your system:

```bash
python3 quick_test_slic.py
```

Expected output:
```
TEST 1: Backbone Token Output       ✓
TEST 2: SLICTransformer Forward     ✓
TEST 3: Full DINO-SLIC Model (E2E)  ✓
ALL TESTS PASSED ✓
```

### Pre-Computing SLIC Maps

DINO-SLIC operates optimally using precomputed `.npz` caches of superpixels during dataloading. Ensure your dataset API is configured to yield `targets['slic_maps'][level]`. If absent, SLIC safely falls back to executing slow unaccelerated CPU segmentation on-the-fly via `skimage`.

### Training

To begin training on a COCO-format dataset (e.g., FASDD):

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

Restore a checkpoint and run test evaluations:
```bash
python3 main.py \
  --output_dir outputs/DINO-SLIC/run1 \
  -c config/DINO/DINO_4scale_slic.py \
  --coco_path /path/to/your/dataset \
  --eval --resume outputs/DINO-SLIC/run1/checkpoint.pth
```

---

## ⚙️ Configuration Parameters

Adjustments available inside `config/DINO/DINO_4scale_slic.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `'slic'` | Directs the model to employ the SLIC Feature pipeline. |
| `slic_n_segments` | `[400, 200, 100]` | Superpixel count hierarchy (fine → coarse structure). |
| `hidden_dim` | `256` | Dimensionality of projected queries and memory tokens. |
| `enc_layers` | `6` | Standard Attention Encoder depth. |
| `dec_layers` | `6` | Standard Attention Decoder depth. |
| `num_queries` | `900` | Maximum candidate detection boxes proposed. |
| `two_stage_type` | `'standard'` | Triggers purely dynamic query formulation via Encoder Centroids. |

---

## 📚 References & Credits

Built as a lightweight analytical counterpart to the primary DINO architecture:

> **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection**
> Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum
> [arXiv:2203.03605](https://arxiv.org/abs/2203.03605)

Original DINO is licensed under the Apache 2.0 license.
