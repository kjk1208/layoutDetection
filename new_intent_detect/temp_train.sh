export CUDA_VISIBLE_DEVICES=6
CUDA_VISIBLE_DEVICES=6 python -u main.py \
  --dataset_root $DATASET_ROOT \
  --dataset pku \
  --batch_size 24 \
  --learning_rate 1e-3 \
  --model_dm_act "none" \
  --model_type "crossattn_encoder_decoder" \
  --epoch 101 \
  --test_interval 20 \
  --checkpoint_interval 20
