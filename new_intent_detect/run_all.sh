#!/bin/bash
set -e # Exit on any error
cd "$(dirname "$0")" # Change to script directory

# Check if DATASET_ROOT is set
if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT environment variable is not set!"
    echo "Please run: export DATASET_ROOT=/path/to/your/dataset"
    exit 1
fi

# Default values
DATASET=${1:-"pku"}
LEARNING_RATES=${2:-"1e-3,1e-4,1e-5,1e-6"}
# Accept multiple activations as comma-separated (e.g., "relu,sigmoid,none")
MODEL_DM_ACTS=${3:-"relu"}
MODEL_TYPE=${4:-"design_intent_detector"}
EPOCHS=${5:-"101"}
BATCH_SIZE=${6:-"16"}
TEST_INTERVAL=${7:-"20"}
CHECKPOINT_INTERVAL=${8:-"20"}
SKIP_TRAINING=${9:-"false"}
CUDA_DEVICE=${10:-"6"}

# Parse learning rates and activations
IFS=',' read -ra LR_ARRAY <<< "$LEARNING_RATES"
IFS=',' read -ra ACT_ARRAY <<< "$MODEL_DM_ACTS"

echo "=========================================="
echo "COMPLETE PIPELINE EXECUTION"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Learning Rates: $LEARNING_RATES"
echo "Model Activations: $MODEL_DM_ACTS"
echo "Model Type: $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Test Interval: $TEST_INTERVAL"
echo "Checkpoint Interval: $CHECKPOINT_INTERVAL"
echo "Skip Training: $SKIP_TRAINING"
echo "CUDA Device: $CUDA_DEVICE"
echo "=========================================="

# Function to run single experiment
run_single_experiment() {
    local LEARNING_RATE="$1"
    local MODEL_DM_ACT="$2"
    # Remove any spaces and convert to safe filename format
    local SAFE_LEARNING_RATE=$(echo "$LEARNING_RATE" | tr -d ' ' | sed 's/[^a-zA-Z0-9._-]/_/g')
    local EXP_NAME="${DATASET}_${BATCH_SIZE}_${SAFE_LEARNING_RATE}_${MODEL_DM_ACT}_${MODEL_TYPE}"
    local LAST_EPOCH=$((EPOCHS - 1))
    
    # Set paths
    if [ $EPOCHS -eq 1 ]; then
        local CKPT_PATH="$(pwd)/${EXP_NAME}/ckpt/epoch0.pth"
        local PRED_DIR="$(pwd)/${EXP_NAME}/result/epoch0/test"
    else
        local CKPT_PATH="$(pwd)/${EXP_NAME}/ckpt/epoch${LAST_EPOCH}.pth"
        local PRED_DIR="$(pwd)/${EXP_NAME}/result/epoch${LAST_EPOCH}/test"
    fi
    local GT_DIR="/home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/${DATASET}/image/test/closedm"
    
    echo ""
    echo "=========================================="
    echo "RUNNING EXPERIMENT: $EXP_NAME"
    echo "Learning Rate: $LEARNING_RATE"
    echo "Activation: $MODEL_DM_ACT"
    echo "=========================================="

    # Step 1: Training (optional)
    if [ "$SKIP_TRAINING" != "true" ]; then
        echo ""
        echo "STEP 1: TRAINING"
        echo "=================="
        echo "Starting training with $EPOCHS epochs..."

        # Create a temporary train script with specific parameters
        cat > temp_train.sh << EOF
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u main.py \\
  --dataset_root \$DATASET_ROOT \\
  --dataset $DATASET \\
  --batch_size $BATCH_SIZE \\
  --learning_rate $LEARNING_RATE \\
  --model_dm_act "$MODEL_DM_ACT" \\
  --model_type "$MODEL_TYPE" \\
  --epoch $EPOCHS \\
  --test_interval $TEST_INTERVAL \\
  --checkpoint_interval $CHECKPOINT_INTERVAL
EOF

        chmod +x temp_train.sh
        source temp_train.sh
        TRAIN_EXIT_CODE=$?

        if [ $TRAIN_EXIT_CODE -ne 0 ]; then
            echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
            rm -f temp_train.sh 2>/dev/null || true
            return 1
        fi

        echo "Training completed successfully!"
        rm -f temp_train.sh 2>/dev/null || true
    else
        echo ""
        echo "STEP 1: TRAINING (SKIPPED)"
        echo "==========================="
        echo "Training step skipped as requested."
        
        # Check if checkpoint exists
        if [ ! -f "$CKPT_PATH" ]; then
            echo "ERROR: Checkpoint file does not exist: $CKPT_PATH"
            echo "Please ensure the model has been trained or provide the correct checkpoint path."
            return 1
        fi
        echo "Using existing checkpoint: $CKPT_PATH"
    fi

    # Step 2: Testing/Inference
    echo ""
    echo "STEP 2: TESTING/INFERENCE"
    echo "=========================="
    echo "Starting inference on test and train sets..."

    # Create a temporary test script
    cat > temp_test.sh << EOF
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
SPLIT=("test" "train")
INFER_CSV=("\${SPLIT[@]}")
DATASET=$DATASET
INFER_CKPT="$CKPT_PATH"
MODEL_DM_ACT=$MODEL_DM_ACT
MODEL_TYPE=$MODEL_TYPE
BATCH_SIZE=$BATCH_SIZE

echo "[DEBUG] model_dm_act: \$MODEL_DM_ACT"
echo "[DEBUG] model_type: \$MODEL_TYPE"
echo "[DEBUG] cuda_device: $CUDA_DEVICE"
echo "Inference on \$DATASET with \$INFER_CKPT"

# maps (single GPU)
for i in {0..1}; do
    python main.py \\
        --dataset_root \$DATASET_ROOT --dataset \$DATASET \\
        --infer --infer_ckpt \$INFER_CKPT \\
        --infer_csv "\${INFER_CSV[i]}" \\
        --model_dm_act=\$MODEL_DM_ACT \\
        --model_type=\$MODEL_TYPE \\
        --batch_size \$BATCH_SIZE
done

# features (single GPU)
for i in {0..1}; do
    python main.py \\
        --dataset_root \$DATASET_ROOT --dataset \$DATASET \\
        --extract --extract_split "\${SPLIT[i]}" --infer_csv "\${INFER_CSV[i]}" \\
        --infer --infer_ckpt \$INFER_CKPT \\
        --model_dm_act=\$MODEL_DM_ACT \\
        --model_type=\$MODEL_TYPE \\
        --batch_size \$BATCH_SIZE
done

# map2box
python map2box.py \\
    --dataset_root \$DATASET_ROOT \\
    --dataset \$DATASET \\
    --infer_ckpt \$INFER_CKPT \\
    --kernel_n 37
EOF

    chmod +x temp_test.sh
    source temp_test.sh
    TEST_EXIT_CODE=$?

    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Testing failed with exit code $TEST_EXIT_CODE"
        rm -f temp_test.sh 2>/dev/null || true
        return 1
    fi

    echo "Testing completed successfully!"
    rm -f temp_test.sh 2>/dev/null || true

    # Step 3: Evaluation
    echo ""
    echo "STEP 3: EVALUATION"
    echo "==================="
    echo "Starting segmentation evaluation..."

    # Check if prediction directory exists
    if [ ! -d "$PRED_DIR" ]; then
        echo "ERROR: Prediction directory does not exist: $PRED_DIR"
        return 1
    fi

    # Check if GT directory exists
    if [ ! -d "$GT_DIR" ]; then
        echo "ERROR: GT directory does not exist: $GT_DIR"
        return 1
    fi

    echo "Prediction directory: $PRED_DIR"
    echo "GT directory: $GT_DIR"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py \
        --pred_dir "$PRED_DIR" \
        --gt_dir "$GT_DIR" \
        --threshold 0.5 \
        --resize_method "bilinear"

    EVAL_EXIT_CODE=$?

    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
        return 1
    fi

    echo "Evaluation completed successfully!"

    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETED: $EXP_NAME"
    echo "=========================================="
    echo "Results saved in:"
    echo "  - Checkpoints: ${EXP_NAME}/ckpt/"
    echo "  - Predictions: ${EXP_NAME}/result/epoch${LAST_EPOCH}/"
    echo "  - Evaluation: ${EXP_NAME}/result/epoch${LAST_EPOCH}/eval/"
    echo "=========================================="
}

# Run experiments for each activation and learning rate
TOTAL_EXPERIMENTS=$(( ${#ACT_ARRAY[@]} * ${#LR_ARRAY[@]} ))
CURRENT_EXPERIMENT=0

for MODEL_DM_ACT in "${ACT_ARRAY[@]}"; do
    for LEARNING_RATE in "${LR_ARRAY[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        echo ""
        echo "=========================================="
        echo "EXPERIMENT $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "=========================================="
        
        run_single_experiment "$LEARNING_RATE" "$MODEL_DM_ACT"
        EXPERIMENT_EXIT_CODE=$?
        
        if [ $EXPERIMENT_EXIT_CODE -ne 0 ]; then
            echo "ERROR: Experiment $CURRENT_EXPERIMENT failed!"
            exit 1
        fi
    done
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Learning rates tested: $LEARNING_RATES"
echo "=========================================="