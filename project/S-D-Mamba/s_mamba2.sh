export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 1024 \
  --learning_rate 0.0001 \
  --d_ff 128 \
  --itr 1 \
  --d_model 64 \
  --expand 2 \
  --d_conv 4 \
  --head_dim 16