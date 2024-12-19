import numpy as np
import matplotlib.pyplot as plt

loaded_timesteps = np.load('sac_ant_1000000_timesteps.npy')[1:]
loaded_rewards = np.load('sac_ant_1000000_rewards.npy')[1:]
# loaded_timesteps = np.load("ppo_timesteps_mean_1000000.npy")#[1:]
# loaded_rewards = np.load("ppo_rewards_mean_1000000.npy")#[1:]
# loaded_timesteps = np.load('ddpg_ant_1000000_timesteps.npy')
# loaded_rewards = np.load('ddpg_ant_1000000_rewards.npy')

filtered_timesteps = []
filtered_rewards = []
# Parameters
window_size = 10  # Number of surrounding rewards to consider
threshold = 0.5   # Maximum allowable deviation (e.g., 2x the average)

for i, (timestep, reward) in enumerate(zip(loaded_timesteps, loaded_rewards)):
    if i==0:
        filtered_timesteps.append(timestep)
        filtered_rewards.append(reward)
    else:
        if reward-filtered_rewards[-1]>-100:
            filtered_timesteps.append(timestep)
            filtered_rewards.append(reward)



plt.figure(figsize=(10, 6))
plt.plot(filtered_timesteps, filtered_rewards, label='Reward', color='blue')
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.title('Reward Over Timesteps')
plt.legend()
plt.grid()
plt.show()
