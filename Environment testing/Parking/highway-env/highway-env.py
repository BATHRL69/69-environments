"""
THIS DOSEN'T WORK
"""

import gym
import highway_env
import numpy as np

from stable_baselines import HER, SAC, DDPG
from stable_baselines.ddpg import NormalActionNoise
env = gym.make("parking-v0")
model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))
# Train for 1e5 steps
model.learn(int(1e5))
# Save the trained agent
model.save('her_sac_highway')
model = HER.load('her_sac_highway', env=env)
obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
	action, _ = model.predict(obs)
	obs, reward, done, info = env.step(action)
	episode_reward += reward
	if done or info.get('is_success', False):
		print("Reward:", episode_reward, "Success?", info.get('is_success', False))
		episode_reward = 0.0
		obs = env.reset()
# Create the action noise object that will be used for exploration
n_actions = env.action_space.shape[0]
noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

model = HER('MlpPolicy', env, DDPG, n_sampled_goal=4,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            actor_lr=1e-3, critic_lr=1e-3, action_noise=action_noise,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))
model.learn(int(2e5))

model.save('her_ddpg_highway')
obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
	action, _ = model.predict(obs)
	obs, reward, done, info = env.step(action)
	episode_reward += reward
	if done or info.get('is_success', False):
		print("Reward:", episode_reward, "Success?", info.get('is_success', False))
		episode_reward = 0.0
		obs = env.reset()