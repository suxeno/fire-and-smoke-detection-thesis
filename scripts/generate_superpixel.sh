#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Move to the parent directory (Project Root)
cd "$SCRIPT_DIR/.."

# Run the superpixel generation script
nohup python3 util/generate_superpixel.py --config configs/detr_slic.yaml > generate_superpixel.out 2>&1 &
PID=$!
echo "Superpixel generation started with PID: $PID"
echo "Logs are being written to generate_superpixel.out"
