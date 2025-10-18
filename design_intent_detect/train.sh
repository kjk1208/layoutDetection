export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET=$1
if [ "$DATASET" = "pku" ]; then
    EPOCH=101
elif [ "$DATASET" = "cgl" ]; then
    EPOCH=36
elif [ "$DATASET" = "all" ]; then
    EPOCH=26
fi

echo "Training on $DATASET with $EPOCH epochs"

torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size 64 --learning_rate 1e-3 --model_dm_act "relu" \
        --epoch $EPOCH

torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size 64 --learning_rate 5e-5 --model_dm_act "relu" \
        --epoch $EPOCH

torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size 64 --learning_rate 3e-5 --model_dm_act "relu" \
        --epoch $EPOCH

torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size 64 --learning_rate 1e-5 --model_dm_act "none" \
        --epoch $EPOCH

torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size 64 --learning_rate 1e-5 --model_dm_act "sigmoid" \
        --epoch $EPOCH