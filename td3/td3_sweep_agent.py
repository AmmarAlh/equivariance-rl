import wandb
import subprocess
import numpy as np

# Read the sweep ID from the file
with open("sweep_id.txt", "r") as f:
    sweep_id = f.read().strip()

# List of seeds for running each experiment
seed_list = [2, 4, 6, 8, 10]  # Define your seeds

PROJECT_NAME = "Equivariant_TD3_InvertedPendulum"


def train(config=None):
    with wandb.init(config=config, project=PROJECT_NAME):
        config = wandb.config  
        cumulative_avg_returns = []  
        
        for seed in seed_list:
            print(f"Running TD3 algorithm with seed {seed}")

            # Run your TD3 algorithm using subprocess for each seed
            result = subprocess.run(
                [
                    "python", "td3/td3_symmetrizer.py",
                    "--wandb-project-name", str(PROJECT_NAME),
                    "--seed", str(seed),
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
                    "--total-timesteps", "5500",
                    "--n-envs", "1",
                    "--no-evaluate",
                ],
                capture_output=True,
                text=True
            )

            print("Errors:", result.stderr)

            # Capture the final cumulative average return from the output
            if "Final Cumulative Avg Return" in result.stdout:
                lines = result.stdout.split("\n")
                for line in lines:
                    if "Final Cumulative Avg Return" in line:
                        avg_return = float(line.split(":")[-1].strip())
                        cumulative_avg_returns.append(avg_return)
                        print(f"Seed {seed}: Cumulative Avg Return = {avg_return}")
                        break
            else:
                print(f"Warning: Cumulative Avg Return not found for seed {seed}")

        # Calculate the average cumulative return across all seeds
        if cumulative_avg_returns:
            avg_cumulative_return = np.mean(cumulative_avg_returns)
            print(f"Final Average Cumulative Return across seeds: {avg_cumulative_return}")

            # Log the final average cumulative return to wandb
            wandb.log({'charts/cumulative_avg_return_ac_seeds': avg_cumulative_return})
        else:
            print("No cumulative average returns recorded.")

        # Finish wandb run
        wandb.finish()

# Launch the sweep agent to run multiple experiments
wandb.agent(sweep_id, project=PROJECT_NAME, function=train, count=2)
