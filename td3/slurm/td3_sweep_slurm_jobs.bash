#!/bin/bash
#SBATCH --job-name=ppo_equi_1               # Job name
#SBATCH --partition=main                      # Partition
#SBATCH --output=logs-slurm/td3/slurm_%j.out  # SLURM and script STDOUT
#SBATCH --error=logs-slurm/td3/slurm_%j.err   # SLURM and script STDERR
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --ntasks=1                            # Run 1 task at a time
#SBATCH --cpus-per-task=8                    # Allocate CPUs per task (adjust based on your needs)

# Show actual node in output file, useful for diagnostics
echo "Starting job on $(hostname) at $(date)"
echo "Running my analysis..."

# Load the necessary modules
source /etc/profile.d/modules.sh
module load nvidia/cuda-12.4
module load anaconda3/2023.09
source activate
# Activate the Conda environment
conda init bash
source ~/.bashrc 
conda activate equivariance-rl

export WANDB__SERVICE_WAIT=300
#/home/s2657708/.conda/envs/equivariance-rl/bin/python td3/td3_optuna.py
#/home/s2657708/.conda/envs/equivariance-rl/bin/wandb agent ammaralh/Equivariant_TD3_InvertedPendulum/8si6nlqw
# echo which python
python -c "import sys; print(f'Using Python interpreter at: {sys.executable}')"
python td3/td3_sweep_agent.py