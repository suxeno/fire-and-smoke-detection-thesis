#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Move to the parent directory (Project Root)
cd "$SCRIPT_DIR/.."

nohup python3 main.py --config configs/detr_hybrid.yaml > train.out 2>&1 &
PID=$!
echo "Training started with PID: $PID"
echo "Logs are being written to train.out"