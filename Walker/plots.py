import numpy as np
import matplotlib.pyplot as plt

loaded_timesteps = np.load("ppo_timesteps_mean.npy")[1:]
loaded_rewards = np.load("ppo_rewards_mean.npy")[1:]
# loaded_timesteps = np.load('sac_timesteps_mean.npy')[1:]
# loaded_rewards = np.load('sac_rewards_mean.npy')[1:]

plt.figure(figsize=(10, 6))
plt.plot(loaded_timesteps, loaded_rewards, label="Reward", color="blue")
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.title("Reward Over Timesteps")
plt.legend()
plt.grid()
plt.show()
