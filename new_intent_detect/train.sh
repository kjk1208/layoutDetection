export CUDA_VISIBLE_DEVICES=6
DATASET=$1
MODEL_TYPE=${2:-design_intent_detector}  # 기본값: design_intent_detector
BATCH_SIZE=32

if [ "$DATASET" = "pku" ]; then
    EPOCH=101
elif [ "$DATASET" = "cgl" ]; then
    EPOCH=36
elif [ "$DATASET" = "all" ]; then
    EPOCH=26
fi


echo "Training on $DATASET with $EPOCH epochs"

# CUDA_VISIBLE_DEVICES=6 python -u main.py \
#   --dataset_root $DATASET_ROOT \
#   --dataset $1 \
#   --batch_size $BATCH_SIZE \
#   --learning_rate 1e-3 \
#   --model_dm_act "relu" \
#   --epoch $EPOCH \

# CUDA_VISIBLE_DEVICES=6 python -u main.py  \
#   --dataset_root $DATASET_ROOT \
#   --dataset $1 \
#   --batch_size $BATCH_SIZE \
#   --learning_rate 1e-4 \
#   --model_dm_act "relu" \
#   --epoch $EPOCH \

CUDA_VISIBLE_DEVICES=6 python -u main.py  \
  --dataset_root $DATASET_ROOT \
  --dataset $1 \
  --batch_size $BATCH_SIZE \
  --learning_rate 1e-5 \
  --model_dm_act "relu" \
  --model_type $MODEL_TYPE \
  --epoch $EPOCH \

CUDA_VISIBLE_DEVICES=6 python -u main.py  \
  --dataset_root $DATASET_ROOT \
  --dataset $1 \
  --batch_size $BATCH_SIZE \
  --learning_rate 1e-3 \
  --model_dm_act "none" \
  --model_type $MODEL_TYPE \
  --epoch $EPOCH \


# torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py \
#         --dataset_root $DATASET_ROOT --dataset $1 \
#         --batch_size 64 --learning_rate 1e-3 --model_dm_act "relu" \
#         --epoch $EPOCH

# torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py \
#         --dataset_root $DATASET_ROOT --dataset $1 \
#         --batch_size 64 --learning_rate 1e-4 --model_dm_act "relu" \
#         --epoch $EPOCH

# torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py \
#         --dataset_root $DATASET_ROOT --dataset $1 \
#         --batch_size 64 --learning_rate 5e-5 --model_dm_act "relu" \
#         --epoch $EPOCH

# torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py \
#         --dataset_root $DATASET_ROOT --dataset $1 \
#         --batch_size 64 --learning_rate 1e-3 --model_dm_act "none" \
#         --epoch $EPOCH

# torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py \
#         --dataset_root $DATASET_ROOT --dataset $1 \
#         --batch_size 64 --learning_rate 1e-3 --model_dm_act "sigmoid" \
#         --epoch $EPOCH