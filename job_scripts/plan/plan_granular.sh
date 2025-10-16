#!/bin/bash

#SBATCH --account=beig-delta-gpu
#SBATCH --partition=gpuA100x4

### NODE/CPU/MEM/GPU ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00

### Misc ###
#SBATCH --ntasks=1

### LOG INFO ###
#SBATCH --job-name=plan_granular
#SBATCH --output=logs/plan/granular/plan_granular_%A-%a.log

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sdayaneni/.mujoco/mujoco210/bin
export C_PATH=:$HOME/miniconda3/envs/dino_wm/include
export DATASET_DIR=/work/nvme/beig/sdayaneni/datasets
module purge
echo "Starting hydra run"

python plan.py model_name=2025-10-14/14-15-13 planner=mpc_cem goal_source='random_state' ckpt_base_path=/work/nvme/beig/sdayaneni/dino_wm