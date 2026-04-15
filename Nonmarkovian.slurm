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
# DFM enhancers (after tar extract to ./data_dfm): use --enhancer_dfm_root data_dfm or auto

# --- Routed model (DiT + Boltzmann router) + DFM enhancer pickle (./data_dfm) ---
# python -m nonmarkovian.train --enhancer_dfm_root data_dfm --enhancer_split train \
#   --max_len 200 --batch_size 8 --epochs 5 --num_classes 0 --save checkpoints/routed_dfm.pt

# --- Routed model (DiT + Boltzmann router) + validation + FBD (FBCNN) ---
# python -m nonmarkovian.train --data data/your.txt --val_split 0.1 --val_fbd_n 256 \
#   --fbcnn_ckpt FBCNN.ckpt --max_len 256 --save checkpoints/routed.pt \
#   --d_model 256 --nhead 8 --dec_layers 8 --dim_ff 1024 --cond_dim 256 \
#   --router_tau 1.0 --router_lambda_bal 0.01

# --- Simple baseline (DiT, no router) + same validation ---
# python -m nonmarkovian.train_simple --data data/your.txt --val_split 0.1 --val_fbd_n 256 \
#   --fbcnn_ckpt FBCNN.ckpt --max_len 256 --save checkpoints/simple_discrete.pt \
#   --d_model 256 --nhead 8 --dec_layers 8 --dim_ff 1024 --cond_dim 256 \
#   --wandb_project nonmarkovian-simple

jupyter notebook --no-browser --port=8889 --ip=0.0.0.0

my-job-stats -a -n -s
