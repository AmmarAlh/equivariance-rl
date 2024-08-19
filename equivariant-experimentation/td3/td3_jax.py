# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import os
import random
import time
from dataclasses import dataclass
import json # for optuna finetunning 

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from emlp.groups import SO, C, D, Trivial
from emlp.nn.flax import EMLPBlock, Linear, Sequential, uniform_rep
from emlp.reps import Scalar, Vector

from env_setup import make_env
from eval import evaluate
from equi_utils import (
    ReacherAngularActionRep,
    InvertedPendulumActionRep,
    equivariance_err_actor,
    equivariance_err_qvalue,
)


os.environ["MUJOCO_GL"] = "egl"

# Determine the base directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the parent output directory
output_dir = os.path.join(base_dir, "output")

# Paths for wandb, runs, and models directories inside the output directory
wandb_dir = os.path.join(output_dir, "")
runs_dir = os.path.join(output_dir, "runs")
models_dir = os.path.join(output_dir, "models")

# Create directories if they do not exist
os.makedirs(wandb_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 4
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_mode: str = "online"
    """the mode of Weights and Biases"""
    wandb_project_name: str = "EquiInvertedPendulum-v4"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `output/runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "InvertedPendulum-v4"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
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
    noise_clip: float = 0.4
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    ch: int = 128
    """the number of channels in the hidden layers of non- and equivaraint networks"""
    # EMLP specific arguments
    use_emlp: bool = False
    """whether to use EMLP for the network architecture"""
    emlp_group: str = "C2"
    """the group of the EMLP layer"""
    # Expert data population of the replay buffer
    use_expert_data: bool = False
    """whether to populate the replay buffer with expert episodes or random samples"""
    expert_actor_model_path: str = (
        "/home/ammaral/Projects/equivaraince-rl/equivariant-experimentation/td3/output/models/InvertedPendulum-v4__td3_jax__1__1723896022__None/td3_jax.cleanrl_model"
    )
    """the path to the expert actor model"""
    evaluate: bool = False
    """whether to evaluate the model"""
    optimizer: str = "adam"
    """the optimizer to use"""
    # Early stopping
    early_stopping: bool = True
    """whether to use early stopping"""
    patience: int = 1500
    """number of steps to wait for improvement before stopping"""
    min_delta: float = 1e-3
    """minimum change in the monitored metric to qualify as an improvement"""
    rolling_window: int = 100
    """number of episodes to consider for rolling average"""


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    ch: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.ch)(x)
        x = nn.relu(x)
        x = nn.Dense(self.ch)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class InvariantQNetwork(nn.Module):
    rep_in: Callable
    rep_out: Callable
    group: Callable
    ch: int = 128

    @nn.compact
    def __call__(self, x, a):
        rep_in = self.rep_in(self.group)
        rep_out = self.rep_out(self.group)
        middle_layers = uniform_rep(self.ch, self.group)

        x = jnp.concatenate([x, a], axis=1)

        network = Sequential(
            EMLPBlock(rep_in=rep_in, rep_out=middle_layers),
            EMLPBlock(rep_in=middle_layers, rep_out=middle_layers),
            Linear(middle_layers, rep_out),
        )

        return network(x)


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    ch: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.ch)(x)
        x = nn.relu(x)
        x = nn.Dense(self.ch)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class EquiActor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    rep_in: Callable
    rep_out: Callable
    group: Callable
    ch: int = 128

    @nn.compact
    def __call__(self, x):
        rep_in = self.rep_in(self.group)
        rep_out = self.rep_out(self.group)
        middle_layers = uniform_rep(self.ch, self.group)

        fc_mu = Sequential(
            EMLPBlock(rep_in=rep_in, rep_out=middle_layers),
            EMLPBlock(rep_in=middle_layers, rep_out=middle_layers),
            Linear(middle_layers, rep_out),
        )

        x = jax.nn.tanh(fc_mu(x))
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


@jax.jit
def update_critic(
    actor_state: TrainState,
    qf1_state: TrainState,
    qf2_state: TrainState,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    terminations: np.ndarray,
    key: jnp.ndarray,
):
    # TODO Maybe pre-generate a lot of random keys
    # also check https://jax.readthedocs.io/en/latest/jax.random.html
    key, noise_key = jax.random.split(key, 2)
    clipped_noise = (
        jnp.clip(
            (jax.random.normal(noise_key, actions.shape) * args.policy_noise),
            -args.noise_clip,
            args.noise_clip,
        )
        * actor.action_scale
    )
    next_state_actions = jnp.clip(
        actor.apply(actor_state.target_params, next_observations) + clipped_noise,
        envs.single_action_space.low,
        envs.single_action_space.high,
    )
    qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
    qf2_next_target = qf.apply(qf2_state.target_params, next_observations, next_state_actions).reshape(-1)
    min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
    next_q_value = (rewards + (1 - terminations) * args.gamma * (min_qf_next_target)).reshape(-1)

    def mse_loss(params):
        qf_a_values = qf.apply(params, observations, actions).squeeze()
        return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
    (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
    qf1_state = qf1_state.apply_gradients(grads=grads1)
    qf2_state = qf2_state.apply_gradients(grads=grads2)

    return (
        (qf1_state, qf2_state),
        (qf1_loss_value, qf2_loss_value),
        (qf1_a_values, qf2_a_values),
        key,
    )


############################ END OF UPDATE_CRITIC ############################
@jax.jit
def update_actor(
    actor_state: TrainState,
    qf1_state: TrainState,
    qf2_state: TrainState,
    observations: np.ndarray,
):
    def actor_loss(params):
        return -qf.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau))

    qf1_state = qf1_state.replace(target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau))
    qf2_state = qf2_state.replace(target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, args.tau))
    return actor_state, (qf1_state, qf2_state), actor_loss_value

    ############################ END OF UPDATE_ACTOR ############################


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ <= "1.2.0":
        raise ValueError("Please install stable-baselines3>=1.2.0 to run this script")

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{args.emlp_group if args.use_emlp else None}"

    if args.track:
        import wandb
        os.environ['WANDB_MODE'] = args.wandb_mode
        wandb.init(
            project=args.wandb_project_name,
            dir=wandb_dir,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    print("Args: ")
    print(args)
    writer = SummaryWriter(f"{runs_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, expert_actor_key, qf1_key, qf2_key = jax.random.split(key, 5)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                0,
                args.capture_video,
                run_name,
                video_path=output_dir,
            )
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float64
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # Init Optimizer
    if args.optimizer == "adam":
        optim = optax.adam(args.learning_rate)
    elif args.optimizer == "sgd":
        optim = optax.sgd(args.learning_rate)

    if args.use_emlp:
        group_mapping = {
            "C2": C(2),
            "C4": C(4),
            "C8": C(8),
            "D4": D(4),
            "SO2": SO(2),
            "Trivial": Trivial(2),
        }
        if args.emlp_group not in group_mapping:
            raise ValueError(f"Unknown group: {args.emlp_group}")
        G = group_mapping[args.emlp_group]

        # Create the state and action representations
        if args.env_id == "Reacher-v4":

            repin_actor = Vector(G) + Vector(G) + Vector(G) + ReacherAngularActionRep(G) + 2 * Scalar(G) + Scalar(G)
            repout_actor = ReacherAngularActionRep(G)

            repin_q = Vector(G) + Vector(G) + Vector(G) + ReacherAngularActionRep(G) + 2 * Scalar(G) + Scalar(G) + repout_actor
            repout_q = Scalar(G)
        elif args.env_id == "InvertedPendulum-v4":
            if G != C(2):
                raise ValueError("InvertedPendulum-v4 only supports C2 group")
            repin_actor = Vector(G) + Vector(G)
            repout_actor = InvertedPendulumActionRep(G)

            repin_q = Vector(G) + Vector(G) + repout_actor
            repout_q = Scalar(G)

        actor = EquiActor(
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
            action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
            rep_in=repin_actor,
            rep_out=repout_actor,
            group=G,
            ch=args.ch,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, obs),
            target_params=actor.init(actor_key, obs),
            tx=optim,
        )

        qf = InvariantQNetwork(rep_in=repin_q, rep_out=repout_q, group=G, ch=args.ch)
        qf1_state = TrainState.create(
            apply_fn=qf.apply,
            params=qf.init(qf1_key, obs, envs.action_space.sample()),
            target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
            tx=optim,
        )
        qf2_state = TrainState.create(
            apply_fn=qf.apply,
            params=qf.init(qf2_key, obs, envs.action_space.sample()),
            target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
            tx=optim,
        )

    else:
        actor = Actor(
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
            action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
            ch=args.ch,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, obs),
            target_params=actor.init(actor_key, obs),
            tx=optim,
        )
        qf = QNetwork(ch=args.ch)
        qf1_state = TrainState.create(
            apply_fn=qf.apply,
            params=qf.init(qf1_key, obs, envs.action_space.sample()),
            target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
            tx=optim,
        )
        qf2_state = TrainState.create(
            apply_fn=qf.apply,
            params=qf.init(qf2_key, obs, envs.action_space.sample()),
            target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
            tx=optim,
        )
    if args.use_expert_data:
        expert_actor = Actor(
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
            action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
            ch=args.ch,
        )

        expert_actor_params = expert_actor.init(actor_key, obs)

        with open(args.expert_actor_model_path, "rb") as f:  # type: ignore
            (expert_actor_params, _, _) = flax.serialization.from_bytes((expert_actor_params, None, None), f.read())

        expert_actor.apply = jax.jit(expert_actor.apply)

    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    start_time = time.time()
    best_avg_return = -np.inf
    steps_without_improvement = 0
    recent_returns = []
    total_reward = 0.0
    total_episodes = 0
    cumulative_avg_return = 0.0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if args.use_expert_data and global_step < args.learning_starts:
            actions = expert_actor.apply(expert_actor_params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(
                            0,
                            max_action * args.exploration_noise,
                            size=envs.single_action_space.shape,
                        )
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )
        elif not args.use_expert_data and global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(
                            0,
                            max_action * args.exploration_noise,
                            size=envs.single_action_space.shape,
                        )
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        if "final_info" in infos:
            for info in infos["final_info"]:
                episodic_return = info["episode"]["r"]
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                episodic_length = info["episode"]["l"]
                writer.add_scalar("charts/epsidic_length", episodic_length, global_step)
 
                # Calculate total reward and total episodes
                total_reward += episodic_return
                total_episodes += 1
                # Calculate cumulative average return
                cumulative_avg_return = (total_reward / total_episodes)
                writer.add_scalar("charts/cumulative_avg_return", cumulative_avg_return, global_step)
                
                # Calculate rolling average of episodic returns
                recent_returns.append(episodic_return)
                if len(recent_returns) > args.rolling_window:
                    recent_returns.pop(0)
                avg_return = np.mean(recent_returns)
                writer.add_scalar("charts/moving_avg_return", avg_return, global_step)
                writer.add_scalar("charts/steps_without_improvement", steps_without_improvement, global_step)
                
                # Early Stopping Check
                if args.early_stopping:
                    if avg_return > best_avg_return + args.min_delta:
                        best_avg_return = avg_return
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += 1
                break
        if steps_without_improvement > args.patience:
            print(f"Early stopping triggered at step {global_step}")
            break
        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
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

            (
                (qf1_state, qf2_state),
                (qf1_loss_value, qf2_loss_value),
                (qf1_a_values, qf2_a_values),
                key,
            ) = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                key,
            )

            if global_step % args.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)

                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                # Calculate and log the equivariance error
                if args.use_emlp:
                    actor_equiv_error = equivariance_err_actor(actor, actor_state.params, obs, repin_actor, repout_actor, G)
                    qf_equiv_error = equivariance_err_qvalue(qf, qf1_state.params, obs, actions, repin_q, repout_q, G)
                    writer.add_scalar("equivariance/actor_error", actor_equiv_error, global_step)
                    writer.add_scalar("equivariance/qf_error", qf_equiv_error, global_step)

    if args.save_model:

        model_path = f"{models_dir}/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                        qf2_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")
        if args.evaluate:
            episodic_returns_eval = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                video_path=output_dir,
                Model=EquiActor if args.use_emlp else Actor,
                exploration_noise=args.exploration_noise,
                seed=args.seed,
                use_emlp=args.use_emlp,
                repin_actor=repin_actor if args.use_emlp else None,
                repout_actor=repout_actor if args.use_emlp else None,
                G=G if args.use_emlp else None,
                ch=args.ch,
            )
            for idx, episodic_return in enumerate(episodic_returns_eval):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)
    
    cumulative_avg_return = float(cumulative_avg_return[0])
    result = {"cumulative_avg_return": cumulative_avg_return}
    print(f"Type of cumulative_avg_return: {type(cumulative_avg_return)}")
    print(f"Value of cumulative_avg_return: {cumulative_avg_return}")
    # Save the results to a JSON file for optuna finetunning
    output_file = f"cumulative_avg_return.json"
    with open(output_file, "w") as f:
        json.dump(result, f)
        
    time.sleep(3)  # prevent
    envs.close()
    writer.close()
