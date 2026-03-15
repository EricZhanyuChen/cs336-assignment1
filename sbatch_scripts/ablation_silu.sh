#!/bin/bash
#SBATCH --job-name=cs336_ablation_silu
#SBATCH --output=logs/ablation_silu_%j.out
#SBATCH --error=logs/ablation_silu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

module purge
module load CUDA/11.8.0
source $HOME/.local/bin/env

cd /home5/s6398820/projects/cs336/assignment1

# Ablation: SWiGLU vs SiLU (d_ff=2048 to match param count)
echo "====================================================="
echo "  SiLU Ablation: lr=1e-3 (d_ff=2048, ~same params)"
echo "====================================================="
TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/train_silu.py \
    --train_data_path data/tinystories_train.npy \
    --val_data_path data/tinystories_val.npy \
    --ckpt_save_path /scratch/s6398820/cs336/checkpoints/ablation_silu_lr1e-3 \
    --device cuda \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 2048 \
    --num_layers 4 \
    --num_heads 16 \
    --rope_theta 10000.0 \
    --batch_size 32 \
    --total_steps 40000 \
    --val_every 500 \
    --save_steps 5000 \
    --lr_max 1e-3 \
    --lr_min 1e-5 \
    --warmup_iters 500 \
    --cosine_iter_cycle 5000 \
    --wandb_name "ablation_silu_lr1e-3"

echo "🎉 SiLU ablation completed!"
