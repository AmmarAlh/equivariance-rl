# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from emlp.nn import uniform_rep
from emlp.reps import Rep, Scalar, Vector, T
from emlp.groups import SO, D, C, O, Trivial
import emlp.nn.pytorch as eqnn
from emlp.nn.pytorch import EMLPBlock, Linear

# Determine the base directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the parent output directory
output_dir = os.path.join(base_dir, 'output')

# Paths for wandb, runs, and models directories inside the output directory
wandb_dir = os.path.join(output_dir, '')
runs_dir = os.path.join(output_dir, 'runs')
models_dir = os.path.join(output_dir, 'models')

# Create directories if they do not exist
os.makedirs(wandb_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


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
    wandb_project_name: str = "equivaraince-rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Reacher-v4"
    """the environment id of the task"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    use_emlp: bool = False
    """use emlp for the Q and actor networks"""
    group: str = "C4"
    """the group to use (C4, C8, D4, SO2, O2)"""
    save_model: bool = True
    """if toggled, the model will be saved at the end of the training"""
    
class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), "This wrapper only works with Box observation spaces"
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = env.observation_space

    def observation(self, observation):
        obs = observation.copy()
        obs[1], obs[2] = obs[2], obs[1]
        # Create the new observation
        #new_obs = np.array([obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[8], obs[9]], dtype=np.float32)
        return obs


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = CustomObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def save_actor(run_name, actor):
    name = sanitize_run_name(run_name)
    model_path = os.path.join(models_dir, f"{name}_actor.pth")
    torch.save(actor.state_dict(), model_path)
    print(f"model saved to {model_path}")
    artifact = wandb.Artifact(name, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

def load_actor(run_name, actor):
    artifact = wandb.use_artifact(f"{run_name}:latest", type="model")
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, f"{run_name}_actor.pth")
    actor.load_state_dict(torch.load(model_path))

import re

def sanitize_run_name(run_name):
    return re.sub(r'[^a-zA-Z0-9-_\.]', '_', run_name)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class InvariantSoftQNetwork(nn.Module):
    def __init__(self, env, rep_in, group, ch=256):
        super().__init__()
        self.rep_in = rep_in(group)
        middle_layers = uniform_rep(ch, group)
        self.G = group
        self.network = nn.Sequential(
            EMLPBlock(rep_in= self.rep_in , rep_out=middle_layers),
            EMLPBlock(rep_in=middle_layers, rep_out=middle_layers),
            Linear(middle_layers, Scalar(group)))

    def forward(self, state, action):
        # Encode state equivariantly
        x = torch.cat([state, action], dim=1)
        return self.network(x)

LOG_STD_MAX = 1
LOG_STD_MIN = -4

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class EquiActor(nn.Module):
    def __init__(self, env, rep_in, rep_out, group, ch=256):
        super().__init__()
        self.G = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        middle_layers = uniform_rep(ch, group)
        # Define equivariant layers using EMLPBlock
        self.network = nn.Sequential(
            EMLPBlock(rep_in=self.rep_in, rep_out=middle_layers),
            EMLPBlock(rep_in=middle_layers, rep_out=middle_layers))
        
        # Final layers for mean and log_std
        self.fc_mean = Linear(middle_layers, self.rep_out)
        self.fc_logstd = Linear(middle_layers, self.rep_out)
        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.network(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "1.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    if args.use_emlp:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__equi={args.use_emlp}__group={args.group}"
    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__equi={args.use_emlp}"
    # Print the parsed arguments
    print("Parsed arguments:", args)
    # Map the group argument to the actual group object
    group_mapping = {
        "C4": C(4),
        "C8": C(8),
        "D4": D(4),
        "SO2": SO(2),
        "D2": D(2),
        "Trivial": Trivial(2)
    }
    if args.group not in group_mapping:
        raise ValueError(f"Unknown group: {args.group}")
    G = group_mapping[args.group]
    # Create the state and action representations
    state_rep = Vector(G) + 2 * Scalar(G) + Vector(G) + 2 * Scalar(G) + Vector(G) + Scalar(G) 
    action_rep = 2*Scalar(G)

    state_rep_q = Vector(G) + 2 * Scalar(G) + Vector(G) + 2 * Scalar(G) + Vector(G) + Scalar(G) + 2*Scalar(G) 
    action_rep_q = Scalar(G)


    if args.track:
        import wandb
        os.environ['WANDB_MODE'] = 'online'  # Set wandb to offline mode

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=wandb_dir,  # Set wandb directory
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    if args.use_emlp:
        actor = EquiActor(envs, state_rep, action_rep, G, ch=256).to(device)
        print(actor)
        print("the number of parameters")
        print(sum(p.numel() for p in actor.parameters()))	
    else:
        actor = Actor(envs).to(device)
        print(actor)
        print("the number of parameters")
        print(sum(p.numel() for p in actor.parameters()))	
    
    if args.use_emlp:
        qf1 = InvariantSoftQNetwork(envs, state_rep_q, G, ch=256).to(device=device)
        print(qf1)
        print("the number of parameters")
        print(sum(p.numel() for p in qf1.parameters()))	    
        qf2 = InvariantSoftQNetwork(envs, state_rep_q, G, ch=256).to(device=device)    
        qf1_target = InvariantSoftQNetwork(envs, state_rep_q, G, ch=256).to(device=device)    
        qf2_target = InvariantSoftQNetwork(envs, state_rep_q, G, ch=256).to(device=device)    
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    else:
        qf1 = SoftQNetwork(envs).to(device)
        print(qf1)
        print("the number of parameters")
        print(sum(p.numel() for p in qf1.parameters()))	  
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
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
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
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

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
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
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
    if args.save_model:
        save_actor(run_name, actor)
    envs.close()
    writer.close()
