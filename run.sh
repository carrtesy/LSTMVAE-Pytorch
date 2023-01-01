echo "running LSTMVAE @gpu $1"
export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --dataset NeurIPS-TS-MUL\
  --hidden_dim 32 \
  --z_dim 3 \
  --n_layers 2\
  --beta 0.01