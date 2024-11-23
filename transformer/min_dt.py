import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import csv
import random
from datetime import datetime
import tyro
from typing import Optional, Dict
from dataclasses import dataclass
import time
import math
from torch.utils.tensorboard import SummaryWriter
# =========================== Utility Functions ===========================

def discount_cumsum(x, gamma):
    """
    Compute discounted cumulative sums of rewards.
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def get_normalized_score(score, env_name):
    """
    Normalize the D4RL score for evaluation.
    """
    REF_MAX_SCORE = {
        'invertedpendulum': 1000.0,
    }
    REF_MIN_SCORE = {
        'invertedpendulum': 0.0,
    }
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f"No reference score for {env_key}"
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


# =========================== Dataset Class ===========================


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):
        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # Handle Gymnasium `timeouts`
        for traj in self.trajectories:
            if 'timeouts' in traj:
                traj['terminals'] = np.logical_or(traj['terminals'], traj['timeouts'])

        # calculate min len of traj, state mean and variance
        # and returns_to-go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns-to-go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask

# =========================== Model Definition ===========================

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s

        return state_preds, action_preds, return_preds

# =========================== Training and Evaluation ===========================

def train_decision_transformer(model, log_dir, optimizer, scheduler, data_loader, device, num_updates_per_iter, max_train_iters, eval_fn):
    """
    Train the Decision Transformer with a dataset and log progress.
    """
    start_time = datetime.now().replace(microsecond=0)
    total_samples = 0
    max_d4rl_score = -1.0


    log_csv_path = os.path.join(log_dir, "training_log.csv")
    csv_writer = csv.writer(open(log_csv_path, 'a', newline=''))
    csv_writer.writerow(["duration", "num_updates", "action_loss", "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    # Convert DataLoader into an iterator
    data_loader_iter = iter(data_loader)
    
    for i_train_iter in range(max_train_iters):
        model.train()
        action_losses = []

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_loader_iter)
            except StopIteration:
                data_loader_iter  = iter(data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_loader_iter)
            context_len = states.shape[1]
            total_samples += context_len
            timesteps, states, actions, returns_to_go, traj_mask = (
                timesteps.to(device),
                states.to(device),
                actions.to(device),
                returns_to_go.to(device).unsqueeze(-1),
                traj_mask.to(device),
            )

            _, action_preds, _ = model(
                timesteps=timesteps,
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
            )

            action_preds = action_preds.view(-1, model.act_dim)[traj_mask.view(-1) > 0]
            action_target = actions.view(-1, model.act_dim)[traj_mask.view(-1) > 0]
            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            action_losses.append(action_loss.item())

        eval_results = eval_fn(model)
        eval_avg_reward = eval_results['eval/avg_reward']
        eval_avg_ep_len = eval_results['eval/avg_ep_len']
        eval_d4rl_score = eval_results['eval/d4rl_score']

        duration = str(datetime.now().replace(microsecond=0) - start_time)
        avg_action_loss = np.mean(action_losses)

        print(f"Iteration {i_train_iter + 1}/{max_train_iters}: Avg Loss: {avg_action_loss:.5f}, Eval Reward: {eval_avg_reward:.5f}, D4RL Score: {eval_d4rl_score:.2f}")
        csv_writer.writerow([duration, total_samples, avg_action_loss, eval_avg_reward, eval_avg_ep_len, eval_d4rl_score])
        
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Action Loss", avg_action_loss, total_samples)
        writer.add_scalar("Eval/Average Reward", eval_avg_reward, total_samples)
        writer.add_scalar("Eval/D4RL Score", eval_d4rl_score, total_samples)
        
        if eval_d4rl_score > max_d4rl_score:
            best_model_path = os.path.join(log_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            max_d4rl_score = eval_d4rl_score
            print(f"  Best model saved at {best_model_path}")

    print(f"Training completed in {duration}. Max D4RL Score: {max_d4rl_score:.2f}")
    
    
    
def evaluate_model_on_env(model, env_name, device, context_len, rtg_target, rtg_scale, num_eval_ep=10, max_ep_len=1000, state_mean=None, state_std=None):
    """
    Evaluate the model on the environment.
    """
    env = gym.make(env_name)
    model.eval()
    total_reward = 0
    total_timesteps = 0
    
    timesteps = torch.arange(start=0, end=max_ep_len, step=1)
    timesteps = timesteps.repeat(1, 1).to(device)
    
    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    with torch.no_grad():
        for _ in range(num_eval_ep):
            
                        # zeros place holders
            actions = torch.zeros((1, max_ep_len, act_dim),
                                dtype=torch.float32, device=device)

            states = torch.zeros((1, max_ep_len, state_dim),
                                dtype=torch.float32, device=device)

            rewards_to_go = torch.zeros((1, max_ep_len, 1),
                                dtype=torch.float32, device=device)
            
            # init episode
            running_state,_ = env.reset(seed=args.seed)
            running_reward = 0
            running_rtg = rtg_target / rtg_scale
            
            for t in range(max_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()


                running_state, running_reward, done, truncations, infos  = env.step(act.cpu().numpy())

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward


                if done or truncations:
                    break

    return {
        "eval/avg_reward": total_reward / num_eval_ep,
        "eval/avg_ep_len": total_timesteps / num_eval_ep,
        "eval/d4rl_score": get_normalized_score(total_reward / num_eval_ep, env_name),
    }




# =========================== Configuration ===========================

@dataclass
class Args:
    # Experiment configuration
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 12
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "decision-transformer-experiment"
    wandb_entity: Optional[str] = None

    # Dataset-related parameters
    dataset_path: str = "./transformer/processed_data/InvertedPendulum-v4-stitched.pkl"
    context_len: int = 20


    # Environment and evaluation configuration
    env_id: str = "InvertedPendulum-v4"
    num_eval_ep: int = 10
    max_ep_len: int = 1000
    rtg_target: float = 1000.0
    rtg_scale: float = 1000.0
    # Training configuration
    batch_size: int = 64
    max_train_iters: int = 200
    num_updates_per_iter: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 0.25
    scheduler_warmup_steps: int = 10000

    # Model configuration
    n_blocks: int = 3
    h_dim: int = 128
    n_heads: int = 1
    drop_p: float = 0.1
    max_timestep: int = 4096
    use_emlp: bool = False
    output_dir: str = "output"

    def __post_init__(self):
        os.makedirs("./dt_runs", exist_ok=True)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}_{args.exp_name}_seed_{args.seed}_{int(time.time())}_{args.use_emlp}"

    # create the output directory where the logging of wandb, tensorboard and model checkpoints will be stored
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    
    print(f"output_dir={output_dir}")
    # Paths for wandb, runs, and models directories inside the output directory
    wandb_dir = os.path.join(output_dir, "")
    runs_dir = os.path.join(output_dir, "runs")
    log_dir = os.path.join(output_dir, "logs")
     # Create directories if they do not exist
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if args.track:
        import wandb
    
        os.environ["WANDB_MODE"] = "online"
        
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
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic   

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    min_len = 10**4
    states = []
    for traj in trajectories:
        min_len = min(min_len, traj['observations'].shape[0])
        states.append(traj['observations'])

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    print(args.dataset_path)
    print("num of trajectories in dataset: ", len(trajectories))
    print("minimum trajectory length in dataset: ", min_len)
    print("state mean: ", state_mean.tolist())
    print("state std: ", state_std.tolist())
    
    
    traj_dataset = D4RLTrajectoryDataset(args.dataset_path, args.context_len, args.rtg_scale)
    traj_data_loader = DataLoader(traj_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    state_mean, state_std = traj_dataset.state_mean, traj_dataset.state_std

    # Model initialization
    state_dim = traj_dataset[0][1].shape[-1]
    act_dim = traj_dataset[0][2].shape[-1]
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=args.n_blocks,
        h_dim=args.h_dim,
        context_len=args.context_len,
        n_heads=args.n_heads,
        drop_p=args.drop_p
    ).to("cuda" if args.cuda else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / args.scheduler_warmup_steps, 1))

    # Training
    train_decision_transformer(
        model,
        log_dir,
        optimizer,
        scheduler,
        traj_data_loader,
        "cuda" if args.cuda else "cpu",
        args.num_updates_per_iter,
        args.max_train_iters,
        eval_fn = lambda model: evaluate_model_on_env(
            model, args.env_id, "cuda" if args.cuda else "cpu", args.context_len, args.rtg_target, args.rtg_scale, args.num_eval_ep, args.max_ep_len, state_mean, state_std
        )
    )
