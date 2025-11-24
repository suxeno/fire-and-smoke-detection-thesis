#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming the script is in scripts/, the project root is one level up
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Cleaning trash files from $PROJECT_ROOT..."

# Clean outputs/logs
if [ -d "$PROJECT_ROOT/outputs/logs" ]; then
    echo "Cleaning outputs/logs/..."
    rm -f "$PROJECT_ROOT/outputs/logs/"*
fi

# Clean outputs/plots
if [ -d "$PROJECT_ROOT/outputs/plots" ]; then
    echo "Cleaning outputs/plots/..."
    rm -f "$PROJECT_ROOT/outputs/plots/"*
fi

# Clean outputs/detr_slic (keeping checkpoints)
if [ -d "$PROJECT_ROOT/outputs/detr_slic" ]; then
    echo "Cleaning outputs/detr_slic/..."
    # Delete everything that is NOT a .pth file
    find "$PROJECT_ROOT/outputs/detr_slic" -maxdepth 1 -type f ! -name "*.pth" -delete
    # Also remove test_results directory if it exists
    rm -rf "$PROJECT_ROOT/outputs/detr_slic/test_results"
fi

# Clean train.out
if [ -f "$PROJECT_ROOT/train.out" ]; then
    echo "Removing train.out..."
    rm -f "$PROJECT_ROOT/train.out"
fi

echo "Trash cleaning complete!"
