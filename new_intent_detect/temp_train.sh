export CUDA_VISIBLE_DEVICES=6
CUDA_VISIBLE_DEVICES=6 python -u main.py \
  --dataset_root $DATASET_ROOT \
  --dataset pku \
  --batch_size 28 \
  --learning_rate 1e-4 \
  --model_dm_act "sigmoid" \
  --epoch 101 \
  --test_interval 20 \
  --checkpoint_interval 20
