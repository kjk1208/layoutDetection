#!/bin/bash

# Segmentation Evaluation Script
# Usage: ./run_eval.sh [pred_dir] [gt_dir] [threshold] [resize_method]

# Default values
PRED_DIR=${1:-"pku_16_0.001_relu/result/epoch100/test"}
GT_DIR=${2:-"/home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/image/test/closedm"}
THRESHOLD=${3:-0.5}
RESIZE_METHOD=${4:-"bilinear"}

echo "Running segmentation evaluation..."
echo "Prediction directory: $PRED_DIR"
echo "GT directory: $GT_DIR"
echo "Threshold: $THRESHOLD"
echo "Resize method: $RESIZE_METHOD"
echo ""

CUDA_VISIBLE_DEVICES=6 python eval.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --threshold "$THRESHOLD" \
    --resize_method "$RESIZE_METHOD"

echo ""
echo "Evaluation completed!"
