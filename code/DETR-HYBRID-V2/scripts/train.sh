#!/usr/bin/env bash
# DETR-HYBRID-V2 train script (efficiency-first)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# CONFIGURATION
DATA_PATH="/home/Media/Dataset/FASDD/FASDD_CV"
OUTPUT_DIR="./outputs/3/"
RESUME_URL="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

# Training parameters
EPOCHS=36
BATCH_SIZE=2
NUM_WORKERS=2

# Model architecture & Hybrid config
ENC_LAYERS=6
DEC_LAYERS=6
SLIC_N_SEGMENTS=200
POOLING_TYPE="mean"

# Phase 2: Mixed token fusion
HYBRID_TOKEN_MODE="mixed"

# Efficiency-first pixel-token pruning (keep ratio clamped to [0.6, 0.8] in code)
PIXEL_PRUNE_KEEP_RATIO="0.8"
PIXEL_PRUNE_SCORE_MODE="saliency"   # saliency | feature_norm | counts
PIXEL_PRUNE_W_FEATURE="0.45"
PIXEL_PRUNE_W_COLOR="0.25"
PIXEL_PRUNE_W_TEXTURE="0.20"
PIXEL_PRUNE_W_SIZE="0.10"


# Learning rates
LR=1e-5
LR_BACKBONE=1e-6

# START TRAINING
echo "DETR-HYBRID-V2 Training"
echo "======================================"
echo "Dataset: $DATA_PATH"  
echo "Output:  $OUTPUT_DIR"
echo "Epochs:  $EPOCHS"
echo "Batch:   $BATCH_SIZE"
echo "LR:      $LR (backbone: $LR_BACKBONE)"
echo "Resume:  $RESUME_URL"
echo "SLIC Segments: $SLIC_N_SEGMENTS"
echo "Pooling: $POOLING_TYPE"
echo "Hybrid token mode: $HYBRID_TOKEN_MODE"
echo "Compact superpixel IDs: true"
echo "Require superpixel files: true"
echo "Pixel prune: enabled"
echo "Pixel prune keep ratio: $PIXEL_PRUNE_KEEP_RATIO"
echo "Pixel prune score mode: $PIXEL_PRUNE_SCORE_MODE"
echo "Pixel prune weights (feature/color/texture/size): $PIXEL_PRUNE_W_FEATURE/$PIXEL_PRUNE_W_COLOR/$PIXEL_PRUNE_W_TEXTURE/$PIXEL_PRUNE_W_SIZE"
echo "Efficiency timing: enabled"
echo ""

nohup python3 main.py \
    --coco_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --enc_layers $ENC_LAYERS \
    --dec_layers $DEC_LAYERS \
    --lr $LR \
    --lr_backbone $LR_BACKBONE \
    --slic_n_segments $SLIC_N_SEGMENTS \
    --pooling_type "$POOLING_TYPE" \
    --hybrid_token_mode "$HYBRID_TOKEN_MODE" \
    --compact_superpixel_ids \
    --require_superpixels \
    --pixel_prune \
    --pixel_prune_keep_ratio "$PIXEL_PRUNE_KEEP_RATIO" \
    --pixel_prune_score_mode "$PIXEL_PRUNE_SCORE_MODE" \
    --pixel_prune_w_feature "$PIXEL_PRUNE_W_FEATURE" \
    --pixel_prune_w_color "$PIXEL_PRUNE_W_COLOR" \
    --pixel_prune_w_texture "$PIXEL_PRUNE_W_TEXTURE" \
    --pixel_prune_w_size "$PIXEL_PRUNE_W_SIZE" \
    --eff_timing \
    --resume "$RESUME_URL" \
    --no_aux_loss \
    > train.log 2>&1 &     

echo "Training started with PID: $!"
echo "Logs: train.log"
echo "JSON Log: $OUTPUT_DIR/training_log.json"