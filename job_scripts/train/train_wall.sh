#!/bin/bash

#SBATCH --account=beig-delta-gpu
#SBATCH --partition=gpuA100x4

### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00

### LOG INFO ###
#SBATCH --job-name=train_wall
#SBATCH --output=logs/train/wall/train_wall_%A-%a.log

### Misc ###
#SBATCH --ntasks=1

module purge
echo "Starting hydra run"

python train.py --config-name train.yaml env=wall frameskip=5 num_hist=3