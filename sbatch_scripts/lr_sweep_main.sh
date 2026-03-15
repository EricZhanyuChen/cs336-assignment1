#!/bin/bash
#SBATCH --job-name=cs336_lr_sweep
#SBATCH --output=lr_sweep_%j.out
#SBATCH --error=lr_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                     # TinyStories 内存不需要 64G 那么多
#SBATCH --gres=gpu:a100:1             # 申请 1 块 A100
#SBATCH --time=04:00:00               # 4 个实验，每个约 40 分钟，申请 4 小时绰绰有余
#SBATCH --partition=gpu

# 环境配置
module purge
module load CUDA/11.8.0
source $HOME/.local/bin/env

# 进入项目目录
cd /home5/s6398820/projects/cs336/assignment1

# 定义需要测试的学习率列表 (包含一个大概率发散的 1e-3, 以及常规的 5e-4, 3e-4, 1e-4)
LEARNING_RATES=("1e-3" "5e-4" "3e-4" "1e-4")

for LR in "${LEARNING_RATES[@]}"; do
    echo "====================================================="
    echo "  🚀 Starting training run with learning rate: $LR"
    echo "====================================================="

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
        --lr_max $LR
done

echo "🎉 All learning rate sweep experiments completed!"