#!/bin/bash
cd /home5/s6398820/projects/cs336/assignment1
#SBATCH --job-name=ts_bs_sweep
#SBATCH --output=../logs/ts_bs_%j.out
#SBATCH --error=../logs/ts_bs_%j.err
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --time=08:00:00 --partition=gpu
# COMPLETED: best=bs=64 lr=1.4e-3 val_loss=1.3082 (no best_model.pt saved)
