#!/bin/bash
#SBATCH --job-name=cs336_bs_1
#SBATCH --output=logs/bs_sweep/bs_1_%j.out
#SBATCH --error=logs/bs_sweep/bs_1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

module purge
module load CUDA/11.8.0
source $HOME/.local/bin/env

cd /home5/s6398820/projects/cs336/assignment1

BS=1
LR="1.7e-4"
TARGET_TOKENS=327680000
CONTEXT_LEN=256

TOTAL_STEPS=$(( TARGET_TOKENS / (BS * CONTEXT_LEN) ))
VAL_EVERY=$(( TOTAL_STEPS / 80 ))
SAVE_STEPS=$(( TOTAL_STEPS / 8 ))

TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/train.py \
    --train_data_path data/tinystories_train.npy \
    --val_data_path data/tinystories_val.npy \
    --ckpt_save_path ./checkpoints/bs_sweep_${BS} \
    --device cuda \
    --vocab_size 10000 \
    --context_length $CONTEXT_LEN \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --rope_theta 10000.0 \
    --batch_size $BS \
    --total_steps $TOTAL_STEPS \
    --val_every $VAL_EVERY \
    --save_steps $SAVE_STEPS \
    --lr_max $LR \
    --cosine_iter_cycle $TOTAL_STEPS \
    --wandb_name "bs_sweep_bs${BS}_lr${LR}"