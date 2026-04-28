# DETR-HYBRID-V2

Superpixel-augmented DETR for fire and smoke object detection with **efficiency-first pixel-token pruning**.

This folder keeps the standard DETR head, Hungarian matching, and COCO evaluation, but augments the transformer encoder input with superpixel structure and (optionally) prunes pixel tokens to reduce compute.

## 1. What Is Different From Vanilla DETR?

Compared to baseline DETR, this variant adds a superpixel branch after the backbone feature projection:

1. Backbone features are projected to `hidden_dim` (`input_proj`).
2. Features are pooled per superpixel ID into `N = slic_n_segments` region tokens.
3. Superpixel centroid positional embeddings are generated.
4. Transformer encoder input is controlled by `hybrid_token_mode`:
   - `superpixel`: encoder uses only superpixel tokens.
   - `mixed`: encoder uses concatenated pixel tokens and superpixel tokens.
5. **V2 pruning (optional, `mixed` only):** pixel tokens are reduced by keeping pixels only inside the top-ranked superpixels until the requested keep ratio is reached.

## 2. Pixel-Token Pruning (V2)

When `--pixel_prune` is enabled, the model:

- Scores each superpixel (`--pixel_prune_score_mode`: `saliency`, `feature_norm`, or `counts`).
- Ranks superpixels and selects enough of the top ones so that the number of pixels covered reaches `--pixel_prune_keep_ratio`.
- Keeps pixels belonging to those selected superpixels; other pixel tokens are removed.

Notes:

- `--pixel_prune_keep_ratio` is clamped to **[0.6, 0.8]** in code.
- Pruning is implemented by gathering/padding tokens so the **encoder sequence length actually shrinks** (not just masking).

## 3. Efficiency Metrics (`eff_*`)

The model emits numeric `eff_*` entries in its forward output dict; `engine.py` automatically logs them during train/eval.

Common keys include:

- `eff_pixel_tokens_before`, `eff_pixel_tokens_after`
- `eff_tokens_before`, `eff_tokens_after`, `eff_tokens_ratio`
- `eff_encoder_seq_len_before`, `eff_encoder_seq_len_after`, `eff_encoder_seq_len_ratio`
- `eff_gflops_before`, `eff_gflops_after`, `eff_gflops_ratio` (rough analytic estimate)

Runtime timing (optional):

- Use `--eff_timing` to log `eff_forward_ms` and `eff_imgs_per_s` (and `eff_iter_ms` during training).
- Use `--eff_timing_sync_cuda` for more accurate GPU timings (slower).

For benchmarking parity, baseline DETR in `code/DETR/` also emits `eff_*` metrics (with pruning disabled).

## 4. Repository Layout

```
DETR-HYBRID-V2/
|- main.py                      # Training/eval entrypoint
|- engine.py                    # Train/eval loops
|- quick_test_detr_hybrid.py    # Sanity test with dummy inputs
|- models/
|  |- detr.py                   # Hybrid model and criterion
|  |- transformer.py            # DETR-style transformer
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

## 5. Data Format

### 5.1 COCO-style annotations

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

### 5.2 Superpixel maps

If you enable superpixel usage (`--hybrid_token_mode mixed|superpixel`), this code looks for:

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

## 6. Environment Setup

From repository root:

```bash
pip install -r requirements.txt
```

Optional speed-up for superpixel pooling:

```bash
pip install torch-scatter
```

If `torch-scatter` is unavailable, the model falls back to a slower but functional PyTorch path.

## 7. Quick Start

### 7.1 Train with provided script

```bash
cd code/DETR-HYBRID-V2
bash scripts/train.sh
```

The script launches training with `nohup` and writes logs to `train.log`.

### 7.2 Train manually (foreground)

```bash
cd code/DETR-HYBRID-V2
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
  --pixel_prune \
  --pixel_prune_keep_ratio 0.8 \
  --pixel_prune_score_mode saliency \
  --eff_timing \
  --no_aux_loss
```

### 7.3 Validation-only run

```bash
cd code/DETR-HYBRID-V2
python3 main.py \
  --coco_path /path/to/dataset \
  --output_dir ./outputs/exp1 \
  --resume ./outputs/exp1/checkpoint.pth \
  --eval \
  --slic_n_segments 200 \
  --pooling_type mean \
  --hybrid_token_mode mixed \
  --compact_superpixel_ids \
  --require_superpixels \
  --pixel_prune \
  --eff_timing
```

## 8. Important Training Arguments

| Argument | Purpose | Typical values |
|---|---|---|
| `--slic_n_segments` | Number of superpixel tokens to load/pool | `200` |
| `--pooling_type` | Feature pooling per segment | `mean`, `max`, `both` |
| `--hybrid_token_mode` | Encoder token composition | `mixed`, `superpixel` |
| `--compact_superpixel_ids` | Per-image dense remap of IDs | flag |
| `--require_superpixels` | Fail if superpixel files are missing | flag |
| `--pixel_prune` | Enable pixel-token pruning (mixed mode) | flag |
| `--pixel_prune_keep_ratio` | Target pixel keep ratio (clamped to `[0.6, 0.8]`) | `0.8` |
| `--pixel_prune_score_mode` | Superpixel ranking score | `saliency`, `feature_norm`, `counts` |
| `--pixel_prune_w_feature` | Saliency weight: pooled feature norm | float (default `0.45`) |
| `--pixel_prune_w_color` | Saliency weight: color cue | float (default `0.25`) |
| `--pixel_prune_w_texture` | Saliency weight: texture/intensity cue | float (default `0.20`) |
| `--pixel_prune_w_size` | Saliency weight: size prior | float (default `0.10`) |
| `--eff_timing` | Log forward/throughput metrics | flag |
| `--eff_timing_sync_cuda` | Synchronize CUDA for timing accuracy | flag |

## 9. Logs and Artifacts

During training, outputs are saved in `--output_dir`:

- `checkpoint.pth`: latest checkpoint
- `log.txt`: JSON-lines per epoch
- `training_log.json`: full JSON array of epoch records
- `info.txt`: experiment metadata and hyperparameters
- `eval/latest.pth`: latest COCO evaluation dump
- `plots/`: generated figures

After final epoch, the script attempts test split evaluation and writes:

- `test_log.json`

## 10. Sanity Test

Run a lightweight functional check (build model, load pretrained DETR weights, run dummy forward/loss):

```bash
cd code/DETR-HYBRID-V2
python3 quick_test_detr_hybrid.py --compact_superpixel_ids --pixel_prune
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