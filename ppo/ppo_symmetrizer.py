# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# symmterizer imports
from symmetrizer.nn.modules import BasisLinear

# import utils
from utils.env_setup import make_env
from utils.eval import evaluate_pytorch
from utils.symmetrizer_utils import create_cartpole_actor_representations, create_cartpole_vfunction_representations


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Equivariance-RL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    wandb_mode: str = "online"
    """whther to run wandb in online or offline mode"""
    output_dir: str = "symmetrizer_output"
    """the root directory of the logs"""
    save_model: bool = True
    """whether to save model into the output_dir/runs/{run_name} folder"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 126
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    ch: int = 64
    """the hidden size of the network"""
    optimizer: str = "adam"
    """the optimizer to use"""
    evaluate: bool = True
    """whether to evaluate the model"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # equivariant mlp specific arguments
    use_emlp: bool = True
    """whether to use equivaraint mlp for the network architecture"""
    emlp_group: str = "C2"
    """the group of the EMLP layer"""
    emlp_basis: str = "equivariant"
    """the basis of the EMLP layer"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.ch)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ch, args.ch)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ch, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.ch)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ch, args.ch)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ch, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class EquiAgent(nn.Module):
    def __init__(self, envs, repr_in_a, repr_out_a, repr_in_q, repr_out_q, hidden_sizes, basis="equivariant", gain_type="xavier"):
        super().__init__()
        self.critic = nn.Sequential(
            BasisLinear(1, hidden_sizes, repr_in_q, basis=basis, gain_type=gain_type, n_samples=4096),
            nn.ReLU(),
            BasisLinear(hidden_sizes, hidden_sizes, repr_out_q, basis=basis, gain_type=gain_type, n_samples=4096),
            nn.ReLU(),
            BasisLinear(hidden_sizes, 1, repr_out_q, basis=basis, gain_type=gain_type, n_samples=4096),
        )
        self.actor = nn.Sequential(
            BasisLinear(1, hidden_sizes, repr_in_a, basis=basis, gain_type=gain_type, n_samples=4096),
            nn.ReLU(),
            BasisLinear(hidden_sizes, hidden_sizes, repr_out_a, basis=basis, gain_type=gain_type, n_samples=4096),
            nn.ReLU(),
            BasisLinear(hidden_sizes, 1, repr_out_a, basis=basis, gain_type=gain_type, n_samples=4096),
        )

    def get_value(self, x):
        return self.critic(x.unsqueeze(1)).squeeze(1)

    def get_action_and_value(self, x, action=None):
        x = x.unsqueeze(1)
        logits = self.actor(x).squeeze(1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x).squeeze(1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}_{args.exp_name}_seed_{args.seed}_n-envs_{args.num_envs}_{int(time.time())}_{args.emlp_group if args.use_emlp else 'None'}"

    # create the output directory where the logging of wandb, tensorboard and model checkpoints will be stored
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    print(f"output_dir={output_dir}")
    # Paths for wandb, runs, and models directories inside the output directory
    wandb_dir = os.path.join(output_dir, "")
    runs_dir = os.path.join(output_dir, "runs")
    models_dir = os.path.join(output_dir, "models")

    # Create directories if they do not exist
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(f"{models_dir}/{run_name}", exist_ok=True)

    if args.track:
        import wandb

        os.environ["WANDB_MODE"] = args.wandb_mode

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=wandb_dir,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # RUNTIME ARGS
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print(f"args={args}")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video, run_name, video_path=output_dir) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Group Representations
    if args.env_id == "CartPole-v1":
        repr_in_a, repr_out_a = create_cartpole_actor_representations()
        repr_in_q, repr_out_q = create_cartpole_vfunction_representations()
    else:
        raise NotImplementedError(f"Environment {args.env_id} is not implemented")

    if not args.use_emlp:
        agent = Agent(envs).to(device)
    else:
        agent = EquiAgent(envs, repr_in_a, repr_out_a, repr_in_q, repr_out_q, args.ch, args.emlp_group).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(agent.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # cummulative returns variables
    total_reward = 0.0
    total_episodes = 0
    cumulative_avg_return = 0.0
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        # Calculate total reward and total episodes
                        total_reward += episodic_return
                        total_episodes += 1
                        # Calculate cumulative average return
                        cumulative_avg_return = total_reward / total_episodes
                        writer.add_scalar("charts/cumulative_avg_return", cumulative_avg_return, global_step)
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    if args.save_model:
        model_path = f"{models_dir}/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    time.sleep(4)  # prevent freezing in the end
    envs.close()
    if args.evaluate:
        episodic_returns = evaluate_pytorch(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=100,
            run_name=f"{run_name}-eval",
            Model=Agent if not args.use_emlp else EquiAgent,
            repr_in=repr_in_a if args.use_emlp else None,
            repr_out=repr_out_a if args.use_emlp else None,
            basis=args.emlp_basis if args.use_emlp else None,
            ch=args.ch,
            device=device,
            seed=args.seed,
            video_path=output_dir,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    writer.close()
