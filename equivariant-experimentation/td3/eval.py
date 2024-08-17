from typing import Callable
import gymnasium as gym
import flax
import flax.linen as nn
import jax
import numpy as np

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    video_path: str,
    Model: nn.Module,
    capture_video: bool = True,
    exploration_noise: float = 0.1,
    seed=1,
    use_emlp=False,
    repin_actor=None,
    repout_actor=None,
    G=None,
    emlp_ch=256,
    ch=256,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, video_path)])
    max_action = float(envs.single_action_space.high[0])
    obs, _ = envs.reset()

    Actor = Model
    action_scale = np.array((envs.action_space.high - envs.action_space.low) / 2.0)
    action_bias = np.array((envs.action_space.high + envs.action_space.low) / 2.0)
    if not use_emlp:
        actor = Actor(
            ch = ch,
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=action_scale,
            action_bias=action_bias,
        )
    else:
        actor = Actor(
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=action_scale,
            action_bias=action_bias,
            rep_in=repin_actor,
            rep_out=repout_actor,
            group=G,
            ch=emlp_ch,
        )
    key = jax.random.PRNGKey(seed)
    key, actor_key = jax.random.split(key, 2)
    actor_params = actor.init(actor_key, obs)

    with open(model_path, "rb") as f:
        actor_params, _, _ = flax.serialization.from_bytes(
            (actor_params, None, None), f.read()
        )

    actor.apply = jax.jit(actor.apply)

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions = actor.apply(actor_params, obs)
        actions = np.array(
            [
                (
                    jax.device_get(actions)[0]
                    + np.random.normal(
                        0,
                        max_action * exploration_noise,
                        size=envs.single_action_space.shape,
                    )
                ).clip(envs.single_action_space.low, envs.single_action_space.high)
            ]
        )

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns