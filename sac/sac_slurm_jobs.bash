#!/bin/bash
#SBATCH --job-name=sac_equi                   # Job name
#SBATCH --partition=main                      # Partition
#SBATCH --output=logs-slurm/sac/slurm_%j.out  # SLURM and script STDOUT
#SBATCH --error=logs-slurm/sac/slurm_%j.err   # SLURM and script STDERR

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

# Parameters
seeds=(1 10 20 30 40)
groups=("C4" "C8" "D4" "SO2")

# Function to run the experiment
run_experiment() {
    local use_emlp_flag=$1
    local group=$2
    local seed=$3
    echo "Starting run_experiment with use_emlp_flag=$use_emlp_flag, group=$group, seed=$seed"
    if [ "$use_emlp_flag" = "True" ]; then
        srun --exclusive /home/s2657708/.conda/envs/equivariance-rl/bin/python /home/s2657708/equi-TF/equivaraince-rl/equivariant-experimentation/sac/sac.py --use-emlp --group $group --seed $seed > ./logs-slurm/sac/sac_reacher_equi_${seed}_${group}.log 2>&1
    else
        srun --exclusive /home/s2657708/.conda/envs/equivariance-rl/bin/python /home/s2657708/equi-TF/equivaraince-rl/equivariant-experimentation/sac/sac.py --seed $seed > ./logs-slurm/sac/sac_reacher_${seed}.log 2>&1
    fi
    echo "Completed run_experiment with use_emlp_flag=$use_emlp_flag, group=$group, seed=$seed"
}

export -f run_experiment

Execute the Python script with different configurations
for seed in "${seeds[@]}"; do
    run_experiment False "" $seed
done

for seed in "${seeds[@]}"; do
    for group in "${groups[@]}"; do
        run_experiment True $group $seed
    done
done
