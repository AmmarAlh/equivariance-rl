import optuna
from optuna.samplers import TPESampler
import subprocess
import json

def objective(trial):
    # Sample hyperparameters using Optuna
    batch_size = trial.suggest_int('batch_size', 128, 512)
    ch = trial.suggest_categorical('ch', [64, 128])
    exploration_noise = trial.suggest_float('exploration_noise', 0.05, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 0.001)
    noise_clip = trial.suggest_float('noise_clip', 0.1, 0.8)
    policy_frequency = trial.suggest_int('policy_frequency', 1, 4)
    policy_noise = trial.suggest_float('policy_noise', 0.025, 0.8)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    # Construct the command to run your script with the sampled hyperparameters
    command = [
        "/home/s2657708/.conda/envs/equivariance-rl/bin/python", "td3/td3_jax.py",
        "--batch_size", str(batch_size),
        "--ch", str(ch),
        "--exploration_noise", str(exploration_noise),
        "--learning_rate", str(learning_rate),
        "--noise_clip", str(noise_clip),
        "--policy_frequency", str(policy_frequency),
        "--policy_noise", str(policy_noise),
        "--tau", str(tau),
        "--optimizer", optimizer,
        "--wandb_project_name", "InvertedPendulum-v4",
        "--wandb_mode", "offline",
        "--use_emlp",
    ]

    # Run the script
    subprocess.run(command, check=True)

    # Load the result from the JSON file
    output_file = "cumulative_avg_return.json"  
    with open(output_file, "r") as f:
        result = json.load(f)

    return result["cumulative_avg_return"] 

# Setup and run the Optuna study
study = optuna.create_study(direction="maximize", sampler=TPESampler())
study.optimize(objective, n_trials=50, n_jobs=1)

# Output the best hyperparameters
print("Best hyperparameters: ", study.best_params)
