import wandb

# Define the sweep configuration
sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'charts/cumulative_avg_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0007,
            'max': 0.007
        },
        'optimizer': {
            'distribution': 'categorical',
            'values': ['adam', 'sgd']
        },
        'batch_size': {
            'distribution': 'categorical',
            'values': [128, 256, 512]
        },
        'exploration_noise': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.3
        },
        'noise_clip': {
            'distribution': 'uniform',
            'min': 0.4,
            'max': 0.6
        },
        'policy_frequency': {
            'distribution': 'categorical',
            'values': [2, 3]
        },
        'tau': {
            'distribution': 'uniform',
            'min': 0.002,
            'max': 0.008
        },
        'policy_noise': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.4
        }
    },
}

PROJECT_NAME = "N_Equivariant_TD3_InvertedPendulum"

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)

# Save the sweep ID for the agent script
with open("sweep_id.txt", "w") as f:
    f.write(sweep_id)

print(f"Sweep initialized with ID: {sweep_id}")
