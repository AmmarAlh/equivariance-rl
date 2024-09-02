# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from typing import Optional
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable 

# symmterizer imports
from symmetrizer.nn.modules import BasisLinear
from symmetrizer.nn.modules import BasisLinear
from symmetrizer.ops import GroupRepresentations
from symmetrizer.groups import MatrixRepresentation

# import utils
from utils.env_setup import make_env
from utils.eval import evaluate_pytorch
from utils.symmetrizer_utils import create_inverted_pendulum_actor_representations, create_inverted_pendulum_qfunction_representations, actor_equivariance_mae, q_equivariance_mae

os.environ["MUJOCO_GL"] = "egl"  # to prevent segmentation fault in headless servers

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "equivaraince-rl"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out videos folder)"""
    wandb_mode: str = "online"
    """whther to run wandb in online or offline mode"""
    output_dir: str = "symmetrizer_output"
    """the root directory of the logs"""
    save_model: bool = True
    """whether to save model into the output_dir/runs/{run_name} folder"""
    
    # Algorithm specific arguments
    env_id: str = "InvertedPendulum-v4"
    """the id of the environment"""
    n_envs: int = 25
    """the number of parallel environments"""	
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 6e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    ch: int = 128
    """the hidden size of the network"""
    optimizer: str = "adam"
    """the optimizer to use"""
    evaluate: bool = True
    """whether to evaluate the model"""
    # equivariant mlp specific arguments
    use_emlp: bool = False
    """whether to use equivaraint mlp for the network architecture"""
    emlp_group: str = "C2"
    """the group of the EMLP layer"""
    emlp_basis: str = "equivariant"
    """the basis of the EMLP layer"""

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), args.ch, dtype=torch.float64)
        self.fc2 = nn.Linear(args.ch,  args.ch, dtype=torch.float64)
        self.fc3 = nn.Linear(args.ch, 1, dtype=torch.float64)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), args.ch, dtype=torch.float64)
        self.fc2 = nn.Linear(args.ch, args.ch, dtype=torch.float64)
        self.fc_mu = nn.Linear(args.ch, np.prod(env.single_action_space.shape), dtype=torch.float64)
        # action rescaling
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


# ALGO LOGIC: initialize equivaraint agent here:
class InvariantQNetwork(nn.Module):
    def __init__(self, env, repr_in, repr_out, hidden_sizes, basis="equivariant", gain_type="xavier"):
        super().__init__()
        self.fc1 = BasisLinear(1, hidden_sizes, repr_in, basis=basis, gain_type=gain_type, n_samples=2*4096)
        self.fc2 = BasisLinear(hidden_sizes, hidden_sizes, repr_out, basis=basis, gain_type=gain_type,  n_samples=2*4096)
        self.fc3 = BasisLinear(hidden_sizes, 1, repr_out, basis=basis, gain_type=gain_type,  n_samples=2*4096)

    def forward(self, x, a):
        x, a = x.unsqueeze(1), a.unsqueeze(1)
        x = torch.cat([x, a], 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)


class EquiActor(nn.Module):
    def __init__(self, env, repr_in, repr_out, hidden_size, basis="equivariant", gain_type="xavier"):
        super().__init__()
        self.fc1 = BasisLinear(1, hidden_size, group=repr_in, basis=basis, gain_type=gain_type, bias_init=False,  n_samples=2*4096)
        self.fc2 = BasisLinear(hidden_size, hidden_size, group=repr_out, basis=basis, gain_type=gain_type, bias_init=False,  n_samples=2*4096)
        self.fc_mu = BasisLinear(hidden_size, 1, group=repr_out, basis=basis, gain_type=gain_type, bias_init=False,  n_samples=2*4096)
        
        # action rescaling
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        x = x * self.action_scale + self.action_bias
        x = x.squeeze(1)
        return x


if __name__ == "__main__":

    args = tyro.cli(Args)
    
    seed_info = f"seeds_{args.seed}_{args.seed + args.n_envs - 1}" if args.n_envs > 1 else f"seed_{args.seed}"
    run_name = f"{args.env_id}__{args.exp_name}__{seed_info}__{int(time.time())}__{args.emlp_group if args.use_emlp else 'None'}"
    
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
    writer = SummaryWriter(f"{runs_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # device: automatically set to GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, video_path=output_dir) for i in range(args.n_envs)])

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # Group Representations
    if args.env_id == "InvertedPendulum-v4":
        repr_in,repr_out = create_inverted_pendulum_actor_representations()
        repr_in_q,repr_out_q = create_inverted_pendulum_qfunction_representations()
    
    
    
    
    if not args.use_emlp:
        actor = Actor(envs).to(device)
        qf1 = QNetwork(envs).to(device)
        qf2 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        qf2_target = QNetwork(envs).to(device)
        target_actor = Actor(envs).to(device)
    else:
        actor = EquiActor(envs, repr_in, repr_out, args.ch, args.emlp_basis).to(device)
        qf1 = InvariantQNetwork(envs, repr_in_q, repr_out_q, args.ch, basis = args.emlp_basis).to(device)
        qf2 = InvariantQNetwork(envs, repr_in_q, repr_out_q, args.ch, basis = args.emlp_basis).to(device)
        qf1_target = InvariantQNetwork(envs, repr_in_q, repr_out_q, args.ch, basis = args.emlp_basis).to(device)
        qf2_target = InvariantQNetwork(envs, repr_in_q, repr_out_q, args.ch, basis = args.emlp_basis).to(device)
        target_actor = EquiActor(envs, repr_in, repr_out, args.ch, basis = args.emlp_basis).to(device)
        
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    if args.optimizer == "adam":
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
    else:
        q_optimizer = optim.SGD(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.SGD(list(actor.parameters()), lr=args.learning_rate)
    # replay buffer
    envs.single_observation_space.dtype = np.float64
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.n_envs,
    )
    start_time = time.time()
    # cummulative returns variables
    total_reward = 0.0
    total_episodes = 0
    cumulative_avg_return = 0.0
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device, dtype=torch.float64))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
            # Calculate total reward and total episodes
            total_reward += episodic_return
            total_episodes += 1
            # Calculate cumulative average return
            cumulative_avg_return = (total_reward / total_episodes)
            writer.add_scalar("charts/cumulative_avg_return", cumulative_avg_return, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle final_observation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device, dtype=torch.float64) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(envs.single_action_space.low[0], envs.single_action_space.high[0])
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                
                err_a = actor_equivariance_mae(actor, data.observations, repr_in, repr_out)
                err_q1 = q_equivariance_mae(qf1, data.observations, data.actions, repr_in_q)
                err_q2 = q_equivariance_mae(qf2, data.observations, data.actions, repr_in_q)
                writer.add_scalar("equivariance/actor_equivariance_mae", err_a, global_step)
                writer.add_scalar("equivariance/qf1_equivariance_mae", err_q1, global_step)
                writer.add_scalar("equivariance/qf2_equivariance_mae", err_q2, global_step)
 

    if args.save_model:
        
        model_path = f"{models_dir}/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(actor.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
    if args.evaluate:
        episodic_returns = evaluate_pytorch(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=25,
            run_name=f"{run_name}-eval",
            Model=Actor if not args.use_emlp else EquiActor,
            repr_in=repr_in if args.use_emlp else None,
            repr_out=repr_out if args.use_emlp else None,
            basis=args.emlp_basis if args.use_emlp else None,
            ch=args.ch,
            device=device,
            exploration_noise=args.exploration_noise,
            seed=args.seed,
            video_path=output_dir,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
 
    
    # for finetunnning purposes with wandb sweep and optuna
    import json
    cumulative_avg_return = float(cumulative_avg_return[0])
    result = {"cumulative_avg_return": cumulative_avg_return}
    print(f"cumulative_avg_return={cumulative_avg_return}")
    output_file = f"cumulative_avg_return.json"
    with open(output_file, "w") as f:
        json.dump(result, f)
    
    time.sleep(5) # prevent premature killing of the enviroment
    envs.close()
    writer.close()
