import wandb
import subprocess
import numpy as np

# Read the sweep ID from the file
with open("sweep_id.txt", "r") as f:
    sweep_id = f.read().strip()

PROJECT_NAME = "N_Equivariant_TD3_InvertedPendulum"


def train(config=None):
    with wandb.init(config=config, project=PROJECT_NAME):
        config = wandb.config  
        print(f"Running TD3 algorithm")

        # Run your TD3 algorithm using subprocess for each seed
        result = subprocess.run(
            [
                "python", "td3/td3_symmetrizer.py",
                "--wandb-project-name", str(PROJECT_NAME),
                "--seed", str(1),
                "--learning-rate", str(config.learning_rate),
                "--batch-size", str(config.batch_size),
                "--optimizer", config.optimizer,
                "--exploration-noise", str(config.exploration_noise),
                "--noise-clip", str(config.noise_clip),
                "--policy-frequency", str(config.policy_frequency),
                "--tau", str(config.tau),
                "--policy-noise", str(config.policy_noise),
                "--track",
                "--use-emlp",
                "--total-timesteps", "300000",
                "--n-envs", "1",
                "--ch", "64",
                "--no-evaluate",
            ],
            capture_output=True,
            text=True
        )

        print("Errors:", result.stderr)

        # Finish wandb run
        wandb.finish()

# Launch the sweep agent to run multiple experiments
wandb.agent(sweep_id, project=PROJECT_NAME, function=train, count=50)
