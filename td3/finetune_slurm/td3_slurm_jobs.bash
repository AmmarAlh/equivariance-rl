#!/bin/bash
#SBATCH --job-name=td3_equi                   # Job name
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

# Define the seeds for sequential runs
seeds=(1 2 3 4 5 6 7)

# Run the command sequentially for each seed
for seed_value in "${seeds[@]}"
do
    echo "Running with seed: $seed_value"  # Debugging statement to confirm the seed value
    
    /home/s2657708/.conda/envs/equivariance-rl/bin/python td3/td3_jax.py \
    --batch_size=128 \
    --ch=128 \
    --exploration_noise=0.1616845205660069 \
    --learning_rate=0.0006050915827478229 \
    --learning_starts=2648  \
    --noise_clip=0.19123162935732332 \
    --optimizer=sgd \
    --policy_frequency=3 \
    --policy_noise=0.7895366960729397 \
    --tau=0.008840053734178458 \
    --wandb-mode offline \
    --seed=$seed_value
    echo "Completed run with seed: $seed_value"  # Indicate completion of the seed
done