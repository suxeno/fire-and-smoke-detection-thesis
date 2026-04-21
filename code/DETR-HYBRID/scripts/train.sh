#!/usr/bin/env bash
# DETR-HYBRID train script
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

# Phase 3: Query priors
# QUERY_PRIOR_MODE="superpixel_topk"
QUERY_PRIOR_MODE="superpixel_saliency"
QUERY_PRIOR_STRENGTH="0.5"
QUERY_PRIOR_W_FEATURE="0.45"
QUERY_PRIOR_W_COLOR="0.25"
QUERY_PRIOR_W_TEXTURE="0.20"
QUERY_PRIOR_W_SIZE="0.10"

# Phase 4A: Extra loss terms
QUERY_PRIOR_LOSS_COEF="0.0"
# QUERY_PRIOR_LOSS_COEF="0.05"  # Enable query-prior alignment loss

# Phase 4B: Attention bias in encoder/decoder (disabled here)
ENCODER_ATTN_BIAS_MODE="none"
ENCODER_ATTN_BIAS_STRENGTH="0.0"
DECODER_ATTN_BIAS_MODE="none"
DECODER_ATTN_BIAS_STRENGTH="0.0"
# ENCODER_ATTN_BIAS_MODE="superpixel_penalty"
# ENCODER_ATTN_BIAS_STRENGTH="1.0"
# DECODER_ATTN_BIAS_MODE="superpixel_penalty"
# DECODER_ATTN_BIAS_STRENGTH="1.0"

# Learning rates
LR=1e-5
LR_BACKBONE=1e-6

# START TRAINING
echo "DETR-HYBRID Training"
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
echo "Query prior mode: $QUERY_PRIOR_MODE"
echo "Query prior strength: $QUERY_PRIOR_STRENGTH"
echo "Query prior weights (feature/color/texture/size): $QUERY_PRIOR_W_FEATURE/$QUERY_PRIOR_W_COLOR/$QUERY_PRIOR_W_TEXTURE/$QUERY_PRIOR_W_SIZE"
echo "Query prior loss coef: $QUERY_PRIOR_LOSS_COEF"
echo "Encoder attention bias mode: $ENCODER_ATTN_BIAS_MODE"
echo "Encoder attention bias strength: $ENCODER_ATTN_BIAS_STRENGTH"
echo "Decoder attention bias mode: $DECODER_ATTN_BIAS_MODE"
echo "Decoder attention bias strength: $DECODER_ATTN_BIAS_STRENGTH"
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
    --query_prior_mode "$QUERY_PRIOR_MODE" \
    --query_prior_strength "$QUERY_PRIOR_STRENGTH" \
    --query_prior_w_feature "$QUERY_PRIOR_W_FEATURE" \
    --query_prior_w_color "$QUERY_PRIOR_W_COLOR" \
    --query_prior_w_texture "$QUERY_PRIOR_W_TEXTURE" \
    --query_prior_w_size "$QUERY_PRIOR_W_SIZE" \
    --query_prior_loss_coef "$QUERY_PRIOR_LOSS_COEF" \
    --encoder_attn_bias_mode "$ENCODER_ATTN_BIAS_MODE" \
    --encoder_attn_bias_strength "$ENCODER_ATTN_BIAS_STRENGTH" \
    --decoder_attn_bias_mode "$DECODER_ATTN_BIAS_MODE" \
    --decoder_attn_bias_strength "$DECODER_ATTN_BIAS_STRENGTH" \
    --resume "$RESUME_URL" \
    --no_aux_loss \
    > train-3.log 2>&1 &     

echo "Training started with PID: $!"
echo "Logs: train.log"
echo "JSON Log: $OUTPUT_DIR/training_log.json"