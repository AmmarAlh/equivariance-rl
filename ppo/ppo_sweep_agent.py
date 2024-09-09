import wandb
import subprocess
import numpy as np

# Read the sweep ID from the file
with open("sweep_id.txt", "r") as f:
    sweep_id = f.read().strip()

PROJECT_NAME = "Equivariant_PPO_Cartpole"


def train(config=None):
    with wandb.init(config=config, project=PROJECT_NAME):
        config = wandb.config  
        
        print(f"Running PPO algorithm")

        # Run your PPO algorithm using subprocess
        result = subprocess.run(
            [
                "python", "ppo/ppo_symmetrizer.py",
                "--wandb-project-name", str(PROJECT_NAME),
                "--learning-rate", str(config.learning_rate),
                "--num-steps", str(config.num_steps),
                "--clip-coef", str(config.clip_coef),
                "--ent-coef", str(config.ent_coef),
                "--gae-lambda", str(config.gae_lambda),
                "--vf-coef", str(config.vf_coef),
                "--anneal-lr" if config.anneal_lr else "--no-anneal-lr",
                "--track",
                "--use-emlp",
                "--total-timesteps", "500000",
                "--num-envs", "2",
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
