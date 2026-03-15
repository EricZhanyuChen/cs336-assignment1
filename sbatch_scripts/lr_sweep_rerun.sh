#!/bin/bash
#SBATCH --job-name=cs336_lr_rerun
#SBATCH --output=log/lr/lr_sweep_rerun%j.out
#SBATCH --error=log/lr/lr_sweep_rerun%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# Environment configuration
module purge
module load CUDA/11.8.0
source $HOME/.local/bin/env

# Navigate to project directory
cd /home5/s6398820/projects/cs336/assignment1

# Create log directory only if it does not exist
if [ ! -d "log/lr" ]; then
    mkdir -p log/lr
fi

# Only rerun the failed experiment (1e-3)
LEARNING_RATES=("1e-3")

for LR in "${LEARNING_RATES[@]}"; do
    echo "====================================================="
    echo "  🚀 Rerunning failed experiment with LR: $LR"
    echo "====================================================="

    # Individual log file for this specific LR within log/lr/
    LOG_FILE="log/lr/train_${LR}.log"

    TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/train.py \
        --train_data_path data/tinystories_train.npy \
        --val_data_path data/tinystories_val.npy \
        --ckpt_save_path ./checkpoints/lr_sweep_${LR} \
        --device cuda \
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --d_ff 1344 \
        --num_layers 4 \
        --num_heads 16 \
        --rope_theta 10000.0 \
        --batch_size 32 \
        --total_steps 40000 \
        --val_every 500 \
        --save_steps 5000 \
        --lr_max $LR > "$LOG_FILE" 2>&1
done

echo "🎉 Rerun of failed experiment completed!"