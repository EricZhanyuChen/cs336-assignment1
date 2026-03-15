#!/bin/bash
#SBATCH --job-name=cs336_owt
#SBATCH --output=logs/owt_%j.out
#SBATCH --error=logs/owt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

module purge
module load CUDA/11.8.0
source $HOME/.local/bin/env

cd /home5/s6398820/projects/cs336/assignment1

# Step 1: Convert OWT vocab format (ID->token to token->ID) if needed
echo "=== Converting OWT vocab format ==="
python3 -c "
import json
with open('data/owt_vocab.json') as f:
    v = json.load(f)
# Check format
sample = list(v.items())[0]
if isinstance(sample[0], str) and sample[0].isdigit():
    # Inverted format: {\"0\": \"Ā\"} -> {\"Ā\": 0}
    inverted = {token: int(id_) for id_, token in v.items()}
    with open('data/owt_vocab_fixed.json', 'w') as f:
        json.dump(inverted, f, ensure_ascii=False)
    print(f'Converted {len(inverted)} tokens to owt_vocab_fixed.json')
else:
    print('Format already correct, copying as-is')
    import shutil
    shutil.copy('data/owt_vocab.json', 'data/owt_vocab_fixed.json')
"

# Step 2: Create symlinks with correct format
echo "=== Setting up links ==="
ln -sf owt_vocab_fixed.json data/owt_32k.json
ln -sf owt_merge.txt data/owt_32k.txt

# Step 3: Tokenize OWT data
echo "=== Tokenizing OWT ==="
TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/encode_text.py --file owt

echo "=== Tokenization complete ==="
ls -lh data/owt_train.npy data/owt_val.npy 2>/dev/null

# Step 4: Train on OWT (same arch as TS, bs=64, lr=1e-3)
echo "=== Training on OWT ==="
TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/train.py \
    --train_data_path data/owt_train.npy \
    --val_data_path data/owt_val.npy \
    --ckpt_save_path /scratch/s6398820/cs336/checkpoints/owt_bs64_lr1e-3 \
    --device cuda \
    --vocab_size 32000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --rope_theta 10000.0 \
    --batch_size 64 \
    --total_steps 20480 \
    --val_every 500 \
    --save_steps 5000 \
    --lr_max 1e-3 \
    --lr_min 1e-5 \
    --warmup_iters 500 \
    --cosine_iter_cycle 5000 \
    --weight_decay 0.01 \
    --betas 0.9 0.95 \
    --grad_clipping_norm 1.0 \
    --wandb_name "owt_bs64_lr1e-3"

echo "🎉 OWT training complete!"
