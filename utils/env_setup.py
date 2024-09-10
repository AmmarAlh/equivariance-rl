import gymnasium as gym
import numpy as np

class ReacherObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ReacherObservationWrapper, self).__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        obs = observation.copy()
        obs[1], obs[2] = obs[2], obs[1]
        return obs

def make_env(env_id, seed, idx, capture_video,run_name, video_path = "output"):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"{video_path}/videos/{run_name}")
        else:
            env = gym.make(env_id)
        if env_id == "Reacher-v4":
            env = ReacherObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk
