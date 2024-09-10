import wandb

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'charts/cumulative_avg_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 1e-3,  # Minimum learning rate
            'max': 2e-2  # Maximum learning rate
        },
        'num_steps': {
            'values': [64, 128, 256, 512]  # Number of steps to run the environment
        },
        'clip_coef': {
            'distribution': 'uniform',
            'min': 0.05,  # Minimum clip coefficient
            'max': 0.3  # Maximum clip coefficient
        },
        'ent_coef': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,  # Minimum entropy coefficient
            'max': 0.005  # Maximum entropy coefficient
        },
        'gae_lambda': {
            'distribution': 'uniform',
            'min': 0.9,  # Minimum lambda for GAE
            'max': 1.0  # Maximum lambda for GAE
        },
        'vf_coef': {
            'distribution': 'uniform',
            'min': 0.1,  # Minimum value function coefficient
            'max': 1.0  # Maximum value function coefficient
        },
        'anneal_lr':{
            'values': [True, False]
        },
    }
}


PROJECT_NAME = "Equivariant_PPO_Cartpole"

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)

# Save the sweep ID for the agent script
with open("sweep_id.txt", "w") as f:
    f.write(sweep_id)

print(f"Sweep initialized with ID: {sweep_id}")
