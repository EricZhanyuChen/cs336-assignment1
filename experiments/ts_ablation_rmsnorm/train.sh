#!/bin/bash
cd /home5/s6398820/projects/cs336/assignment1
#SBATCH --job-name=ts_abl_rms
#SBATCH --output=../logs/ts_abl_rms_%j.out
#SBATCH --error=../logs/ts_abl_rms_%j.err
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --time=02:00:00 --partition=gpu
# COMPLETED: val_loss=1.8028
