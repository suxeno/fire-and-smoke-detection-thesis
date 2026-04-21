# DETR-HYBRID

Superpixel-augmented DETR for fire and smoke object detection.

This folder is a DETR variant that injects precomputed SLIC superpixel structure into the transformer pipeline. It keeps the standard DETR detection head and training loop, while adding optional mechanisms for:

- superpixel token pooling
- mixed pixel+superpixel token encoding
- superpixel-guided query initialization
- superpixel-consistency attention bias

The implementation is designed for COCO-style datasets and is adapted for the FASDD layout used in this project.

## 1. What Is Different From Vanilla DETR?

Compared to baseline DETR, this variant adds a superpixel branch after the backbone feature projection:

1. Backbone features are projected to `hidden_dim` (`input_proj`).
2. Features are pooled per superpixel ID into `N = slic_n_segments` region tokens.
3. Superpixel centroid positional embeddings are generated.
4. Transformer input is selected by `hybrid_token_mode`:
   - `superpixel`: encoder uses only superpixel tokens.
   - `mixed`: encoder uses concatenated pixel tokens and superpixel tokens.
5. Decoder can optionally start from a superpixel-based query prior:
  - `superpixel_topk`: rank by superpixel size (token count).
  - `superpixel_saliency`: rank by weighted saliency cues (feature, color, texture/intensity, size prior).
6. Encoder/decoder attention can optionally penalize cross-superpixel attention (`*_attn_bias_mode=superpixel_penalty`).

The detection head, Hungarian matching, and COCO evaluation remain DETR-style.

## 2. Core Features

- Optional superpixel tokenization with configurable pooling (`mean`, `max`, `both`)
- Optional compacting of sparse superpixel IDs to contiguous indices
- Optional query-prior alignment loss (`loss_query_prior_align`)
- Optional attention bias in both encoder and decoder
- Automatic epoch-by-epoch logging to JSON and plot generation
- Automatic test-set evaluation at training end (if test split exists)

## 3. Repository Layout

```
DETR-HYBRID/
|- main.py                      # Training/eval entrypoint
|- engine.py                    # Train/eval loops
|- quick_test_detr_hybrid.py    # Sanity test with dummy inputs
|- models/
|  |- detr.py                   # Hybrid model and criterion
|  |- transformer.py            # Transformer with optional attn biases
|  |- position_encoding.py      # Includes superpixel positional embedding
|- datasets/
|  |- coco.py                   # COCO + superpixel map loading
|  |- transforms.py             # Joint image/box/slic_map transforms
|- util/
|  |- plot_utils.py             # Training plot generation
|- scripts/
|  |- train.sh                  # Example training launcher
|- outputs/                     # Checkpoints, logs, eval artifacts, plots
```

## 4. Data Format

### 4.1 COCO-style annotations

`main.py` expects `--coco_path` and uses `datasets/coco.py` to resolve splits.

Supported layouts:

1. FASDD-style (auto-detected):

```
<coco_path>/
|- images/
|- annotations/
|  |- COCO_*/
|     |- Annotations/
|        |- train.json
|        |- val.json
|        |- test.json
```

2. Standard COCO fallback:

```
<coco_path>/
|- train2017/
|- val2017/
|- annotations/
   |- instances_train2017.json
   |- instances_val2017.json
```

### 4.2 Superpixel maps

If you enable superpixel usage, this code looks for:

```
<coco_path>/superpixels-<N>/<relative_image_parent>/<image_stem>.npz
```

Each `.npz` must contain key:

- `sp_map`: 2D integer array of shape `[H, W]`

Notes:

- `N` must match `--slic_n_segments`.
- Invalid IDs are converted to `-1`.
- With `--compact_superpixel_ids`, IDs are remapped per image to dense `[0..K-1]`.
- With `--require_superpixels`, missing files raise an error.

## 5. Environment Setup

From repository root:

```bash
pip install -r requirements.txt
```

Optional speed-up for superpixel pooling:

```bash
pip install torch-scatter
```

If `torch-scatter` is unavailable, the model falls back to a slower but functional PyTorch path.

## 6. Quick Start

### 6.1 Train with provided script

```bash
cd code/DETR-HYBRID
bash scripts/train.sh
```

The script launches training with `nohup` and writes logs to `train.log`.

### 6.2 Train manually (foreground)

```bash
cd code/DETR-HYBRID
python3 main.py \
  --coco_path /path/to/dataset \
  --output_dir ./outputs/exp1 \
  --epochs 36 \
  --batch_size 2 \
  --num_workers 2 \
  --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
  --slic_n_segments 200 \
  --pooling_type mean \
  --hybrid_token_mode mixed \
  --compact_superpixel_ids \
  --require_superpixels \
  --query_prior_mode superpixel_topk \
  --query_prior_strength 0.5 \
  --query_prior_loss_coef 0.0 \
  --encoder_attn_bias_mode none \
  --decoder_attn_bias_mode none \
  --no_aux_loss
```

### 6.3 Validation-only run

```bash
cd code/DETR-HYBRID
python3 main.py \
  --coco_path /path/to/dataset \
  --output_dir ./outputs/exp1 \
  --resume ./outputs/exp1/checkpoint.pth \
  --eval \
  --slic_n_segments 200 \
  --pooling_type mean \
  --hybrid_token_mode mixed \
  --compact_superpixel_ids \
  --require_superpixels
```

## 7. Important Training Arguments

| Argument | Purpose | Typical values |
|---|---|---|
| `--slic_n_segments` | Number of superpixel tokens to load/pool | `200` |
| `--pooling_type` | Feature pooling per segment | `mean`, `max`, `both` |
| `--hybrid_token_mode` | Encoder token composition | `mixed`, `superpixel` |
| `--compact_superpixel_ids` | Per-image dense remap of IDs | flag |
| `--require_superpixels` | Fail if superpixel files are missing | flag |
| `--query_prior_mode` | Query initialization strategy | `none`, `superpixel_topk`, `superpixel_saliency` |
| `--query_prior_strength` | Scale for injected query prior | float (for example `0.5`) |
| `--query_prior_w_feature` | Saliency weight for pooled feature norm | float (default `0.45`) |
| `--query_prior_w_color` | Saliency weight for color cue | float (default `0.25`) |
| `--query_prior_w_texture` | Saliency weight for texture/intensity cue | float (default `0.20`) |
| `--query_prior_w_size` | Saliency weight for size prior | float (default `0.10`) |
| `--query_prior_loss_coef` | Weight of prior alignment loss | `0.0` to disable, positive to enable |
| `--encoder_attn_bias_mode` | Encoder attention penalty mode | `none`, `superpixel_penalty` |
| `--encoder_attn_bias_strength` | Encoder penalty magnitude | float |
| `--decoder_attn_bias_mode` | Decoder cross-attention penalty mode | `none`, `superpixel_penalty` |
| `--decoder_attn_bias_strength` | Decoder penalty magnitude | float |

## 8. Suggested Phase Progression

The provided `scripts/train.sh` comments imply this staged progression:

1. Phase 1: superpixel pooling + hybrid tokens
2. Phase 2: mixed token fusion (`--hybrid_token_mode mixed`)
3. Phase 3: query priors (`--query_prior_mode superpixel_topk` or `--query_prior_mode superpixel_saliency`)
4. Phase 4A: optional query prior alignment loss
5. Phase 4B: optional encoder/decoder attention bias penalties

This lets you isolate the effect of each mechanism in ablation studies.

Query prior ranking rule (short):

- In `superpixel_topk` mode, superpixels are ranked by `pooled_counts` (how many valid feature-map locations belong to each superpixel after resizing/downsampling).
- In `superpixel_saliency` mode, superpixels are ranked by a weighted score:

$$
score_{sp} = w_f \cdot norm(feature\_norm) + w_c \cdot norm(color\_saliency) + w_t \cdot norm(texture\_intensity) + w_s \cdot norm(\log(1 + pooled\_counts))
$$

where normalization is done per-image over valid superpixels.

`query_prior_strength` is still a single global scaler applied after selecting top-k superpixels and building prior vectors. The `w_*` values only control the ranking step, not the final prior amplitude.

## 9. Logs, Metrics, and Artifacts

During training, outputs are saved in `--output_dir`:

- `checkpoint.pth`: latest checkpoint
- `log.txt`: JSON-lines per epoch
- `training_log.json`: full JSON array of epoch records
- `info.txt`: experiment metadata and hyperparameters
- `eval/latest.pth`: latest COCO evaluation dump
- `plots/`: generated figures
  - `loss_curves.png`
  - `ap_metrics.png`
  - `recall_metrics.png`
  - `timing.png`

After final epoch, the script attempts test split evaluation and writes:

- `test_log.json`

## 10. Sanity Test

Run a lightweight functional check (build model, load pretrained DETR weights, run dummy forward/loss):

```bash
cd code/DETR-HYBRID
python3 quick_test_detr_hybrid.py --compact_superpixel_ids --query_prior_mode superpixel_topk
```

The script calls `model(..., debug=True)` and prints tensor shapes and statistics.

## 11. Known Constraints

- `num_classes` is currently set to `2` in `models/detr.py` for fire/smoke experiments.
- If you adapt to a different dataset, update class mapping and `num_classes` logic accordingly.
- Superpixel generation is not included in this folder; maps must be precomputed offline.

## 12. Troubleshooting

### Missing superpixel directory

- Symptom: warning or `FileNotFoundError` around `superpixels-<N>`.
- Fix: verify `<coco_path>/superpixels-<N>` exists and `N` matches `--slic_n_segments`.

### Slow superpixel pooling

- Symptom: significantly longer epoch time in hybrid mode.
- Fix: install `torch-scatter` to enable the fast pooling path.

### Checkpoint load warnings for class head

- Symptom: shape mismatch warnings when loading pretrained DETR.
- Behavior: expected for transfer learning with different class counts; mismatched keys are filtered automatically.

### Plot generation warnings

- Symptom: warnings about plot utility during training.
- Fix: confirm plotting dependencies are installed and `training_log.json` is valid JSON.

## 13. Credits

This implementation is based on Facebook Research DETR and extends it with superpixel-guided tokenization and attention mechanisms for fire and smoke detection experiments.