import argparse
import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.env_setup import make_env
# symmterizer imports
from symmetrizer.nn.modules import BasisLinear
from utils.symmetrizer_utils import create_inverted_pendulum_actor_representations, create_inverted_pendulum_qfunction_representations, actor_equivariance_mae, q_equivariance_mae

repr_in,repr_out = create_inverted_pendulum_actor_representations()
repr_in_q,repr_out_q = create_inverted_pendulum_qfunction_representations()

#from td3 import Actor  # Ensure this is imported from your TD3 implementation
ch = 64
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), ch, dtype=torch.float32)
        self.fc2 = nn.Linear(ch, ch, dtype=torch.float32)
        self.fc_mu = nn.Linear(ch, np.prod(env.action_space.shape), dtype=torch.float32)
        # action rescaling
        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class EquiActor(nn.Module):
    def __init__(self, env, repr_in, repr_out, hidden_size, basis="equivariant", gain_type="xavier"):
        super().__init__()
        self.fc1 = BasisLinear(1, hidden_size, group=repr_in, basis=basis, gain_type=gain_type, bias_init=False,  n_samples=4096)
        self.fc2 = BasisLinear(hidden_size, hidden_size, group=repr_out, basis=basis, gain_type=gain_type, bias_init=False,  n_samples=4096)
        self.fc_mu = BasisLinear(hidden_size, 1, group=repr_out, basis=basis, gain_type=gain_type, bias_init=False,  n_samples=4096)
        
        # action rescaling
        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        x = x * self.action_scale + self.action_bias
        x = x.squeeze(1)
        return x

def load_td3_actor(model_path, env):
    """
    Load the trained TD3 actor network.
    """
    actor = Actor(env)
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    actor.eval()  # Set the model to evaluation mode
    return actor


def get_reset_data():
    """
    Initialize a dictionary to store trajectory data.
    """
    data = dict(
        observations=[],
        next_observations=[],
        actions=[],
        rewards=[],
        terminals=[],
        timeouts=[],
        logprobs=[],  # Not used in TD3 but kept for compatibility
        # qpos=[],
        # qvel=[]
    )
    return data


def rollout_td3(actor, env_name, max_path, num_data, random=False):
    """
    Generate a dataset by rolling out the TD3 policy in the environment.
    """
    env = gym.make(env_name)
    data = get_reset_data()
    traj_data = get_reset_data()

    _returns = 0
    t = 0
    s, info = env.reset()
    while len(data['rewards']) < num_data:
        if random:
            a = env.action_space.sample()
            logprob = np.log(1.0 / np.prod(env.action_space.high - env.action_space.low))
        else:
            torch_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = actor(torch_s).numpy().reshape(env.action_space.shape)   # Get the action
            logprob = np.nan  # TD3 does not provide log probabilities
        # Mujoco-specific state data
        #qpos, qvel = env.unwrapped.sim.data.qpos.ravel().copy(), env.unwrapped.sim.data.qvel.ravel().copy()

        try:
           # print("action",a,a.shape)
            next_obs, rew, terminated, truncated, info = env.step(a)
        except Exception as e:
            print(f"Environment error: {e}. Resetting environment...")
            env.close()
            env = gym.make(env_name)
            s, info = env.reset()
            traj_data = get_reset_data()
            t = 0
            _returns = 0
            continue

        _returns += rew
        t += 1
        timeout = truncated
        terminal = terminated

        traj_data['observations'].append(s)
        traj_data['actions'].append(a)
        traj_data['next_observations'].append(next_obs)
        traj_data['rewards'].append(rew)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        traj_data['logprobs'].append(logprob)
        # traj_data['qpos'].append(qpos)
        # traj_data['qvel'].append(qvel)

        s = next_obs
        if terminal or timeout:
            print(f'Finished trajectory. Len={t}, Returns={_returns}. Progress: {len(data["rewards"])}/{num_data}')
            s, info = env.reset()
            t = 0
            _returns = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = get_reset_data()

    # Convert collected data to numpy arrays
    new_data = dict(
        observations=np.array(data['observations'], dtype=np.float32),
        actions=np.array(data['actions'], dtype=np.float32),
        next_observations=np.array(data['next_observations'], dtype=np.float32),
        rewards=np.array(data['rewards'], dtype=np.float32),
        terminals=np.array(data['terminals'], dtype=np.bool_),
        timeouts=np.array(data['timeouts'], dtype=np.bool_)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs'], dtype=np.float32)
    # new_data['infos/qpos'] = np.array(data['qpos'], dtype=np.float32)
    # new_data['infos/qvel'] = np.array(data['qvel'], dtype=np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help="Gymnasium environment name")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained TD3 actor model.")
    parser.add_argument('--output_file', type=str, default='output.hdf5', help="Output file path for the offline dataset.")
    parser.add_argument('--max_path', type=int, default=1000, help="Maximum number of steps per trajectory.")
    parser.add_argument('--num_data', type=int, default=10000, help="Total number of samples to collect.")
    parser.add_argument('--random', action='store_true', help="Use random actions instead of the policy.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the environment and the policy
    env = gym.make(args.env)
    policy = None
    if not args.random:
        policy = load_td3_actor(args.model_path, env)

    # Rollout the policy to generate the dataset
    data = rollout_td3(policy, args.env, max_path=args.max_path, num_data=args.num_data, random=args.random)

    # Save the dataset to an HDF5 file
    with h5py.File(args.output_file, 'w') as hfile:
        for k in data:
            hfile.create_dataset(k, data=data[k], compression='gzip')
        hfile['metadata/algorithm'] = np.bytes_('TD3')
        hfile['metadata/policy/nonlinearity'] = np.bytes_('relu')
        hfile['metadata/policy/output_distribution'] = np.bytes_('tanh')

    print(f"Offline dataset saved to {args.output_file}")