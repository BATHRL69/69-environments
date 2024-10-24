# https://gymnasium.farama.org/environments/mujoco/

import gymnasium as gym

env = gym.make("InvertedPendulum-v4", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(10000000000000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()