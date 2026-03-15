#!/bin/bash
#SBATCH --job-name=owt_baseline
#SBATCH --output=../logs/owt_%j.out
#SBATCH --error=../logs/owt_%j.err
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --time=04:00:00 --partition=gpu
# RUNNING: tokenize + train on OWT, bs=64, lr=1e-3
cd /home5/s6398820/projects/cs336/assignment1
module purge; module load CUDA/11.8.0; source $HOME/.local/bin/env
# Vocab fix (if needed)
[ ! -f data/owt_vocab_fixed.json ] && python3 -c "
import json
with open('data/owt_vocab.json') as f: v = json.load(f)
inverted = {token: int(id_) for id_, token in v.items()}
json.dump(inverted, open('data/owt_vocab_fixed.json','w'), ensure_ascii=False)"
ln -sf owt_vocab_fixed.json data/owt_32k.json 2>/dev/null
ln -sf owt_merge.txt data/owt_32k.txt 2>/dev/null
# Tokenize
[ ! -f data/owt_train.npy ] && TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/encode_text.py --file owt
# Train
TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/train.py \
    --train_data_path data/owt_train.npy \
    --val_data_path data/owt_val.npy \
    --ckpt_save_path /scratch/s6398820/cs336/checkpoints/owt_bs64_lr1e-3 \
    --device cuda --vocab_size 32000 --context_length 256 \
    --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 \
    --rope_theta 10000.0 --batch_size 64 --total_steps 20480 \
    --val_every 500 --save_steps 5000 --lr_max 1e-3 --lr_min 1e-5 \
    --warmup_iters 500 --cosine_iter_cycle 5000 --weight_decay 0.01 \
    --betas 0.9 0.95 --grad_clipping_norm 1.0 \
    --wandb_name "owt_bs64_lr1e-3"
