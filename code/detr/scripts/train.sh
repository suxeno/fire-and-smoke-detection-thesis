#!/usr/bin/env bash
# DETR train script
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# CONFIGURATION
DATA_PATH="/home/Media/Dataset/FASDD/FASDD_CV"
OUTPUT_DIR="./outputs/from_scratch"

# Training parameters
EPOCHS=36
BATCH_SIZE=2
NUM_WORKERS=2

# Model architecture
ENC_LAYERS=6
DEC_LAYERS=6

# Learning rates
LR=1e-4
LR_BACKBONE=1e-5

# START TRAINING
echo "DETR Training"
echo "======================================"
echo "Dataset: $DATA_PATH"  
echo "Output:  $OUTPUT_DIR"
echo "Epochs:  $EPOCHS"
echo "Batch:   $BATCH_SIZE"
echo "LR:      $LR (backbone: $LR_BACKBONE)"
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
    --no_aux_loss \
    > train-detr.log 2>&1 &     

echo "Training started with PID: $!"
echo "Logs: train-detr.log"
echo "JSON Log: $OUTPUT_DIR/training_log.json"
