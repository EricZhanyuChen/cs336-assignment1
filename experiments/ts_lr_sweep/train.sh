#!/bin/bash
cd /home5/s6398820/projects/cs336/assignment1
#SBATCH --job-name=ts_lr_sweep
#SBATCH --output=../logs/ts_lr_%j.out
#SBATCH --error=../logs/ts_lr_%j.err
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --time=04:00:00 --partition=gpu
# COMPLETED: lr=1e-3, val_loss=1.5509
