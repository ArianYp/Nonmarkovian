#!/usr/bin/env bash
# Example commands for Nonmarkovian training and sampling.
# Run from this directory (the repo root containing `nonmarkovian/`):
#   cd /path/to/Nonmarkovian
#   bash run_commands.sh          # does nothing by itself — copy/paste sections below
#
# On the cluster (see Nonmarkovian.slurm): module load Python, source venv, then run the python lines.

# set -euo pipefail   # optional; enable if you turn commands below into active runs
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# -----------------------------------------------------------------------------
# Quick dev (DFM pickles under data_dfm; uses pickle train + val)
# -----------------------------------------------------------------------------
# python -m nonmarkovian.train \
#   --dfm_enhancer auto --epochs 2 --batch_size 4 --max_len 500 \
#   --num_timesteps 16 --d_model 64 --nhead 4 --dec_layers 4 --dim_ff 256 \
#   --no-wandb --save checkpoints/routed_dev.pt

# python -m nonmarkovian.train_simple \
#   --dfm_enhancer auto --epochs 2 --batch_size 4 --max_len 500 \
#   --num_timesteps 16 --d_model 64 --nhead 4 --dec_layers 4 --dim_ff 256 \
#   --no-wandb --save checkpoints/simple_dev.pt

# -----------------------------------------------------------------------------
# DFM enhancer pickles (Zenodo 10184648, Taskiran et al. tarball)
# Extract into this repo as: $ROOT/data_dfm/the_code/General/data/DeepFlyBrain_data.pkl
#   curl -L -o Taskiran_et_al_code_models_data.tar.gz "https://zenodo.org/records/10184648/files/Taskiran_et_al_code_models_data.tar.gz?download=1"
#   mkdir -p data_dfm && tar -xzf Taskiran_et_al_code_models_data.tar.gz -C data_dfm
# --dfm_enhancer data_dfm | auto | absolute path   (--dfm_melanoma for DeepMEL2)
# Training always uses pickle train + val (no random split).
# -----------------------------------------------------------------------------
# python -m nonmarkovian.train \
#   --dfm_enhancer "$ROOT/data_dfm" --max_len 500 \
#   --batch_size 8 --epochs 5 --num_classes 0 \
#   --save checkpoints/routed_dfm_fb.pt --no-wandb
# python -m nonmarkovian.train_simple \
#   --dfm_enhancer auto --max_len 500 \
#   --batch_size 8 --epochs 5 --num_classes 0 --save checkpoints/simple_dfm_fb.pt --no-wandb

# -----------------------------------------------------------------------------
# Routed model — DFM enhancer + validation + FBD (FBCNN embeddings, fly brain 81 classes)
# --val_fbd_n is how many sequences feed FBD each epoch (not --max_len). Adjust --dfm_enhancer / --fbcnn_ckpt.
# -----------------------------------------------------------------------------
# python -m nonmarkovian.train \
#   --dfm_enhancer auto \
#   --max_len 500 \
#   --batch_size 8 \
#   --epochs 5 \
#   --lr 3e-4 \
#   --num_timesteps 32 \
#   --d_model 256 \
#   --nhead 8 \
#   --dec_layers 8 \
#   --dim_ff 1024 \
#   --cond_dim 0 \
#   --time_freq_dim 256 \
#   --dropout 0.1 \
#   --router_tau 1.0 \
#   --router_lambda_bal 0.01 \
#   --val_batch_size 8 \
#   --val_fbd_n 500 \
#   --val_gen_batch 8 \
#   --fbcnn_ckpt FBCNN.ckpt \
#   --fbcnn_num_cls 81 \
#   --fbcnn_stacks 4 \
#   --save checkpoints/routed.pt \
#   --wandb_project nonmarkovian \
#   --wandb_run_name routed

# -----------------------------------------------------------------------------
# Simple DiT baseline (no router) — same validation / FBD hooks
# -----------------------------------------------------------------------------
# python -m nonmarkovian.train_simple \
#   --dfm_enhancer auto \
#   --max_len 500 \
#   --batch_size 8 \
#   --epochs 5 \
#   --lr 3e-4 \
#   --num_timesteps 32 \
#   --d_model 256 \
#   --nhead 8 \
#   --dec_layers 8 \
#   --dim_ff 1024 \
#   --cond_dim 0 \
#   --time_freq_dim 256 \
#   --dropout 0.1 \
#   --val_batch_size 8 \
#   --val_fbd_n 500 \
#   --val_gen_batch 8 \
#   --fbcnn_ckpt FBCNN.ckpt \
#   --fbcnn_num_cls 81 \
#   --fbcnn_stacks 4 \
#   --save checkpoints/simple_discrete.pt \
#   --wandb_project nonmarkovian \
#   --wandb_run_name simple_dit

# -----------------------------------------------------------------------------
# Paper-scale transformers (time embedding dim128 everywhere; FFN = 4 × hidden)
#
# 1) Small ~110M:  12 layers, d_model 768,  12 heads, dim_ff 3072
# 2) Medium ~460M: 24 layers, d_model 1024, 16 heads, dim_ff 4096  (lower --batch_size on one GPU)
# 3) Large ~1.7B:  48 layers, d_model 1536, 24 heads,  dim_ff 6144  (multi-GPU / small batch)
# -----------------------------------------------------------------------------

# --- Small (~110M): routed ---
# python -m nonmarkovian.train \
#   --dfm_enhancer auto \
#   --max_len 500 --batch_size 4 --epochs 5 --lr 3e-4 --num_timesteps 32 \
#   --d_model 768 --nhead 12 --dec_layers 12 --dim_ff 3072 \
#   --cond_dim 0 --time_freq_dim 128 --dropout 0.1 \
#   --router_tau 1.0 --router_lambda_bal 0.01 \
#   --val_fbd_n 500 --fbcnn_ckpt FBCNN.ckpt \
#   --save checkpoints/routed_small.pt --wandb_run_name routed_small_110m

# --- Small (~110M): simple DiT ---
# python -m nonmarkovian.train_simple \
#   --dfm_enhancer auto \
#   --max_len 500 --batch_size 4 --epochs 5 --lr 3e-4 --num_timesteps 32 \
#   --d_model 768 --nhead 12 --dec_layers 12 --dim_ff 3072 \
#   --cond_dim 0 --time_freq_dim 128 --dropout 0.1 \
#   --val_fbd_n 500 --fbcnn_ckpt FBCNN.ckpt \
#   --save checkpoints/simple_small.pt --wandb_run_name simple_small_110m

# --- Medium (~460M): routed ---
# python -m nonmarkovian.train \
#   --dfm_enhancer auto \
#   --max_len 500 --batch_size 2 --epochs 5 --lr 3e-4 --num_timesteps 32 \
#   --d_model 1024 --nhead 16 --dec_layers 24 --dim_ff 4096 \
#   --cond_dim 0 --time_freq_dim 128 --dropout 0.1 \
#   --router_tau 1.0 --router_lambda_bal 0.01 \
#   --val_fbd_n 500 --val_batch_size 2 --fbcnn_ckpt FBCNN.ckpt \
#   --save checkpoints/routed_medium.pt --wandb_run_name routed_medium_460m

# --- Medium (~460M): simple DiT ---
# python -m nonmarkovian.train_simple \
#   --dfm_enhancer auto \
#   --max_len 500 --batch_size 2 --epochs 5 --lr 3e-4 --num_timesteps 32 \
#   --d_model 1024 --nhead 16 --dec_layers 24 --dim_ff 4096 \
#   --cond_dim 0 --time_freq_dim 128 --dropout 0.1 \
#   --val_fbd_n 500 --val_batch_size 2 --fbcnn_ckpt FBCNN.ckpt \
#   --save checkpoints/simple_medium.pt --wandb_run_name simple_medium_460m

# --- Large (~1.7B): routed ---
# python -m nonmarkovian.train \
#   --dfm_enhancer auto \
#   --max_len 500 --batch_size 1 --epochs 5 --lr 3e-4 --num_timesteps 32 \
#   --d_model 1536 --nhead 24 --dec_layers 48 --dim_ff 6144 \
#   --cond_dim 0 --time_freq_dim 128 --dropout 0.1 \
#   --router_tau 1.0 --router_lambda_bal 0.01 \
#   --val_fbd_n 500 --val_batch_size 1 --fbcnn_ckpt FBCNN.ckpt \
#   --save checkpoints/routed_large.pt --wandb_run_name routed_large_1p7b

# --- Large (~1.7B): simple DiT ---
# python -m nonmarkovian.train_simple \
#   --dfm_enhancer auto \
#   --max_len 500 --batch_size 1 --epochs 5 --lr 3e-4 --num_timesteps 32 \
#   --d_model 1536 --nhead 24 --dec_layers 48 --dim_ff 6144 \
#   --cond_dim 0 --time_freq_dim 128 --dropout 0.1 \
#   --val_fbd_n 500 --val_batch_size 1 --fbcnn_ckpt FBCNN.ckpt \
#   --save checkpoints/simple_large.pt --wandb_run_name simple_large_1p7b

# -----------------------------------------------------------------------------
# Sampling (after training)
# -----------------------------------------------------------------------------
# python -m nonmarkovian.sample \
#   --checkpoint checkpoints/routed.pt \
#   --batch 8 --seq_len 500 --device cuda --out samples_routed.txt

# python -m nonmarkovian.sample_simple \
#   --checkpoint checkpoints/simple_discrete.pt \
#   --batch 8 --seq_len 500 --device cuda --out samples_simple.txt

# Optional: class-conditional (if trained with --num_classes > 0)
#   --label 0

# -----------------------------------------------------------------------------
# Extras
# -----------------------------------------------------------------------------
# --no-wandb                          Disable Weights & Biases
# --log_timing                        CUDA-synced per-batch timings (train.py / train_simple.py)
# --num_classes N --aux_beta 0.1      Label embedding + aux head (needs labels in dataset)

echo "Reference script: uncomment one block above or copy a command. Repo root: $ROOT"
