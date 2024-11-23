import gymnasium as gym
from stable_baselines3 import SAC, PPO
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Helper functions"))
)

from update_xml import update_env_xml
from render_learning import render_learning

custom_xml_path = update_env_xml(True, False, False)
env = gym.make(
    "Ant-v4",
    render_mode="rgb_array",
    xml_file=custom_xml_path,
)


num_timesteps = 40_000
model = SAC("MlpPolicy", env, verbose=1, device="cuda", train_freq=4)
model.learn(total_timesteps=num_timesteps)
model.save("ppo_inverted_double_pendulum")


obs, info = env.reset()
render_learning(num_timesteps, env, model)
