#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198

module load GCCcore/11.3.0
module load Python/3.11.3

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

source virtualenv/Python3.11.3/bin/activate

# Install dependencies if needed (uncomment for first run):
# pip install -r requirements.txt
# pip install einops  # required for DiT blocks
# pip install flash-attn --no-build-isolation  # optional, falls back to SDPA

# Submit from this directory (Nonmarkovian), same layout as EnergybasedDit/Energy.slurm
# DFM enhancers: --dfm_enhancer data_dfm (or auto)

# --- Routed model (DiT + Boltzmann router) + DFM enhancer pickle (./data_dfm) ---
# python -m nonmarkovian.train --dfm_enhancer data_dfm \
#   --max_len 500 --batch_size 8 --epochs 5 --num_classes 0 --save checkpoints/routed_dfm.pt

# --- Routed model (DiT + Boltzmann router) + DFM + FBD (FBCNN) ---
# python -m nonmarkovian.train --dfm_enhancer data_dfm --val_fbd_n 500 \
#   --fbcnn_ckpt FBCNN.ckpt --max_len 500 --save checkpoints/routed.pt \
#   --d_model 256 --nhead 8 --dec_layers 8 --dim_ff 1024 --cond_dim 256 \
#   --router_tau 1.0 --router_lambda_bal 0.01

# --- Simple baseline (DiT, no router) + DFM + same validation ---
# python -m nonmarkovian.train_simple --dfm_enhancer data_dfm --val_fbd_n 500 \
#   --fbcnn_ckpt FBCNN.ckpt --max_len 500 --save checkpoints/simple_discrete.pt \
#   --d_model 256 --nhead 8 --dec_layers 8 --dim_ff 1024 --cond_dim 256 \
#   --wandb_project nonmarkovian-simple

jupyter notebook --no-browser --port=8889 --ip=0.0.0.0

my-job-stats -a -n -s
