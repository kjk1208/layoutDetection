export CUDA_VISIBLE_DEVICES=6
SPLIT=("test" "train")
INFER_CSV=("${SPLIT[@]}")
DATASET=$1
INFER_CKPT=$2
MODEL_DM_ACT=${3:-none} 
BATCH_SIZE=${4:-32}

echo "[DEBUG] model_dm_act: $MODEL_DM_ACT"
echo "Inferece on $DATASET with $INFER_CKPT"

# maps (single GPU)
for i in {0..1}; do
    # torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
    #     --dataset_root $DATASET_ROOT --dataset $DATASET \
    #     --infer --infer_ckpt $INFER_CKPT \
    #     --infer_csv "${INFER_CSV[i]}" \
    #     --model_dm_act=$MODEL_DM_ACT
    python main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --infer --infer_ckpt $INFER_CKPT \
        --infer_csv "${INFER_CSV[i]}" \
        --model_dm_act=$MODEL_DM_ACT \
        --batch_size $BATCH_SIZE
done

# if [ "$DATASET" = "all" ]; then
#     return 0
# fi

# features (single GPU)
for i in {0..1}; do
# torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
    python main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --extract --extract_split "${SPLIT[i]}" --infer_csv "${INFER_CSV[i]}" \
        --infer --infer_ckpt $INFER_CKPT \
        --model_dm_act=$MODEL_DM_ACT \
        --batch_size $BATCH_SIZE
done

# map2box
python map2box.py \
    --dataset_root $DATASET_ROOT \
    --dataset $DATASET \
    --infer_ckpt $INFER_CKPT \
    --kernel_n 37