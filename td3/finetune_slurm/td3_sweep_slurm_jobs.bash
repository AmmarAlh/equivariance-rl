#!/bin/bash
#SBATCH --job-name=td3-equi-3                 # Job name
#SBATCH --partition=main                      # Partition
#SBATCH --output=logs-slurm/td3/slurm_%j.out  # SLURM and script STDOUT
#SBATCH --error=logs-slurm/td3/slurm_%j.err   # SLURM and script STDERR
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --ntasks=1                            # Run 1 task at a time
#SBATCH --cpus-per-task=4                     # Allocate CPUs per task (adjust based on your needs)

# Show actual node in output file, useful for diagnostics
hostname

# Load the necessary modules
source /etc/profile.d/modules.sh
module load anaconda3/2024.02
module load nvidia/cuda-12.4

# Activate the conda environment
source ~/.bashrc
conda init bash
conda activate equivariance-rl


#/home/s2657708/.conda/envs/equivariance-rl/bin/python td3/td3_optuna.py
/home/s2657708/.conda/envs/equivariance-rl/bin/wandb agent ammaralh/Equivariant_TD3_InvertedPendulum/8si6nlqw