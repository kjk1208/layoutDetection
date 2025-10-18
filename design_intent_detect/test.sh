export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SPLIT=("test" "valid" "train")
INFER_CSV=("test" "train" "train")
DATASET=$1
INFER_CKPT=$2
MODEL_DM_ACT=${3:-none} 

echo "[DEBUG] model_dm_act: $MODEL_DM_ACT"
echo "Inferece on $DATASET with $INFER_CKPT"

# maps
for i in {0..1}; do
    torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --infer --infer_ckpt $INFER_CKPT \
        --infer_csv "${INFER_CSV[i]}" \
        --model_dm_act=$MODEL_DM_ACT
done

if [ "$DATASET" = "all" ]; then
    return 0
fi

# features
for i in {0..2}; do
torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --extract --extract_split "${SPLIT[i]}" --infer_csv "${INFER_CSV[i]}" \
        --infer --infer_ckpt $INFER_CKPT \
        --model_dm_act=$MODEL_DM_ACT
done

# map2box
python map2box.py \
    --dataset_root $DATASET_ROOT \
    --dataset $DATASET \
    --infer_ckpt $INFER_CKPT \
    --kernel_n 37