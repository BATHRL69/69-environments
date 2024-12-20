import numpy as np
import matplotlib.pyplot as plt

# loaded_timesteps = np.load("ppo_timesteps_mean_1000000.npy")#[1:]
# loaded_rewards = np.load("ppo_rewards_mean_1000000.npy")#[1:]

ddpg_timesteps = np.load("ddpg_ant_timesteps_1000000.npy")
ddpg_rewards = np.load("ddpg_ant_rewards_1000000.npy")

td3_timesteps = np.load("td3_ant_timesteps_1000000.npy")
td3_rewards = np.load("td3_ant_rewards_1000000.npy")

sac_timesteps = np.load('sac_ant_1000000_timesteps.npy')[1:]
sac_rewards = np.load('sac_ant_1000000_rewards.npy')[1:]

ppo_timesteps = np.load("ppo_timesteps_mean_1000000.npy")
ppo_rewards = np.load("ppo_rewards_mean_1000000.npy")

sac_filtered_timesteps = []
sac_filtered_rewards = []
for i, (timestep, reward) in enumerate(zip(sac_timesteps, sac_rewards)):
    if i==0:
        sac_filtered_timesteps.append(timestep)
        sac_filtered_rewards.append(reward)
    else:
        if reward-sac_filtered_rewards[-1]>-300:
            sac_filtered_timesteps.append(timestep)
            sac_filtered_rewards.append(reward)

# plt.figure(figsize=(10, 6))
# plt.plot(ddpg_timesteps, ddpg_rewards, label="DDPG", color="blue")
# plt.plot(td3_timesteps, td3_rewards, label="TD3", color="green")
# plt.plot(sac_filtered_timesteps, sac_filtered_rewards, label="SAC", color="red")
# plt.plot(ppo_timesteps, ppo_rewards, label="PPO", color="purple")
# plt.xlabel("Timesteps")
# plt.ylabel("Rewards")
# plt.title("Rewards vs Timesteps for Different Algorithms")
# plt.legend()
# plt.grid()
# plt.show()


def plot(timesteps,rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, label="reward", color="red")
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Timesteps")
    plt.legend()
    plt.grid()
    plt.show()
