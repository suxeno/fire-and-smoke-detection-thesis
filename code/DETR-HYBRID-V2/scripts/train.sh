#!/usr/bin/env bash
# DETR-HYBRID-V2 training script
# Superpixel-guided pixel-token pruning for efficient object detection
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Dataset & output
DATA_PATH="/root/datasets/FASDD/FASDD_CV"
OUTPUT_DIR="./outputs/2-withwarmingepoch/"
RESUME_URL="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

# Training parameters
EPOCHS=36
BATCH_SIZE=4
NUM_WORKERS=10
LR=1e-5
LR_BACKBONE=1e-6

# Model architecture
ENC_LAYERS=6
DEC_LAYERS=6

# Superpixel-guided pruning
SLIC_N_SEGMENTS=200
PIXEL_PRUNE_KEEP_RATIO="0.8"  # fraction of pixel tokens to keep (clamped to [0.6, 0.8])
PIXEL_PRUNE_WARMUP_EPOCHS=6  # train N epochs with full context before enabling pruning

# Score mode determines HOW pixels are ranked for pruning:
#   feature_norm  — L2 norm of backbone features (fast, no extra compute)
#   saliency      — weighted combination of feature norm + fire/smoke color
#                   cues + texture gradients + superpixel size (uses W_* weights below)
#   counts        — raw superpixel pixel counts (simplest)
PIXEL_PRUNE_SCORE_MODE="saliency"
PIXEL_PRUNE_W_FEATURE="0.50"
PIXEL_PRUNE_W_COLOR="0.20"
PIXEL_PRUNE_W_TEXTURE="0.20"
PIXEL_PRUNE_W_SIZE="0.10"

# Print config
echo "DETR-HYBRID-V2 Training"
echo "======================================"
echo "Dataset: $DATA_PATH"
echo "Output:  $OUTPUT_DIR"
echo "Epochs:  $EPOCHS  |  Batch: $BATCH_SIZE  |  Workers: $NUM_WORKERS"
echo "LR:      $LR (backbone: $LR_BACKBONE)"
echo "Resume:  $RESUME_URL"
echo "SLIC segments: $SLIC_N_SEGMENTS"
echo "Pixel prune: keep_ratio=$PIXEL_PRUNE_KEEP_RATIO  score_mode=$PIXEL_PRUNE_SCORE_MODE  warmup=$PIXEL_PRUNE_WARMUP_EPOCHS epochs"
if [[ "$PIXEL_PRUNE_SCORE_MODE" == "saliency" ]]; then
    echo "  Saliency weights: feature=$PIXEL_PRUNE_W_FEATURE color=$PIXEL_PRUNE_W_COLOR texture=$PIXEL_PRUNE_W_TEXTURE size=$PIXEL_PRUNE_W_SIZE"
fi
echo ""

# Build command
CMD=(
    python3 main.py
    --coco_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --epochs $EPOCHS
    --batch_size $BATCH_SIZE
    --num_workers $NUM_WORKERS
    --enc_layers $ENC_LAYERS
    --dec_layers $DEC_LAYERS
    --lr $LR
    --lr_backbone $LR_BACKBONE
    --slic_n_segments $SLIC_N_SEGMENTS
    --require_superpixels
    --pixel_prune
    --pixel_prune_keep_ratio "$PIXEL_PRUNE_KEEP_RATIO"
    --pixel_prune_score_mode "$PIXEL_PRUNE_SCORE_MODE"
    --pixel_prune_warmup_epochs $PIXEL_PRUNE_WARMUP_EPOCHS
    --eff_timing
    --resume "$RESUME_URL"
    --no_aux_loss
)

# Saliency weights only matter when score_mode=saliency
if [[ "$PIXEL_PRUNE_SCORE_MODE" == "saliency" ]]; then
    CMD+=(
        --pixel_prune_w_feature "$PIXEL_PRUNE_W_FEATURE"
        --pixel_prune_w_color "$PIXEL_PRUNE_W_COLOR"
        --pixel_prune_w_texture "$PIXEL_PRUNE_W_TEXTURE"
        --pixel_prune_w_size "$PIXEL_PRUNE_W_SIZE"
    )
fi

nohup "${CMD[@]}" > train.log 2>&1 &

echo "Training started with PID: $!"
echo "Logs: train.log"
echo "JSON Log: $OUTPUT_DIR/training_log.json"