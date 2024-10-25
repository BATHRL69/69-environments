import sys
import os
import time
from collections import deque
import heapq
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from shapely.geometry import LinearRing, LineString, Point
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# Add the local tactics2d directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "./tactics2d/tactics2d"))
)

from tactics2d.envs import ParkingEnv
from tactics2d.math.interpolate import ReedsShepp
from tactics2d.traffic.status import ScenarioStatus

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

## Environment
# the proportion of the type of parking lot,
# 0 means all scenarios are parallel parking, 1 means all scenarios are vertical parking
type_proportion = 1.0
# the render mode, "rgb_array" means render the scene to a numpy array, "human" means render the scene to a window
render_mode = ["rgb_array", "human"][1]
render_fps = 10
# the max step of one episode
max_step = 1000
env = ParkingEnv(
    type_proportion=type_proportion,
    render_mode=render_mode,
    render_fps=render_fps,
    max_step=max_step,
)

# Check if CUDA is available and set the device
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    gamma=0.995,
    learning_rate=2e-6,
    n_steps=20,
    batch_size=32,
    tensorboard_log="./logs",
    device=device,
)

# Train the PPO agent
num_timesteps = 100  # Adjust the number of timesteps as needed
model.learn(total_timesteps=num_timesteps)

# Save the trained model
model.save("ppo_parking")

# Evaluate the PPO agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")


# Define the evaluation function
def eval_rl_agent(env, model, episode_num=100, verbose=True):
    reward_list = deque(maxlen=episode_num)
    success_list = deque(maxlen=episode_num)
    loss_list = deque(maxlen=episode_num)
    status_info = deque(maxlen=episode_num)

    for _ in range(episode_num):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        reward_list.append(total_reward)
        success_list.append(info.get("is_success", False))

    if verbose:
        print(f"Average reward: {np.mean(reward_list)}")
        print(f"Success rate: {np.mean(success_list)}")

    return reward_list, success_list


# Evaluate the trained agent
eval_rl_agent(env, model, episode_num=100)
