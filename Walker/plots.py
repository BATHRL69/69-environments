import numpy as np
import matplotlib.pyplot as plt

# random_timesteps = np.load("random_timesteps_1000000_1.npy")
# random_rewards = np.load("random_rewards_1000000_1.npy")
# random_timesteps_2 = np.load("random_timesteps_1000000_2.npy")
# random_rewards_2 = np.load("random_rewards_1000000_2.npy")
# random_timesteps_3 = np.load("random_timesteps_1000000_3.npy")
# random_rewards_3 = np.load("random_rewards_1000000_3.npy")

# ddpg_timesteps = np.load("ddpg_ant_timesteps_1000000.npy")
# ddpg_rewards = np.load("ddpg_ant_rewards_1000000.npy")
# ddpg_timesteps_2 = np.load("ddpg_ant_timesteps_1000000_2.npy")
# ddpg_rewards_2 = np.load("ddpg_ant_rewards_1000000_2.npy")
# ddpg_timesteps_3 = np.load("ddpg_ant_timesteps_1000000_2.npy")
# ddpg_rewards_3 = np.load("ddpg_ant_rewards_1000000_2.npy")

# td3_timesteps = np.load("td3_ant_timesteps_1000000.npy")
# td3_rewards = np.load("td3_ant_rewards_1000000.npy")
# td3_timesteps_2 = np.load("td3_ant_timesteps_1000000_1.npy")
# td3_rewards_2 = np.load("td3_ant_rewards_1000000_1.npy")
# td3_timesteps_3 = np.load("td3_ant_timesteps_1000000_2.npy")
# td3_rewards_3 = np.load("td3_ant_rewards_1000000_2.npy")

# sac_timesteps = np.load('sac_ant_1000000_timesteps.npy')[1:]
# sac_rewards = np.load('sac_ant_1000000_rewards.npy')[1:]
# sac_timesteps_2 = np.load('sac_ant_timesteps_1000000_2.npy')[1:]
# sac_rewards_2 = np.load('sac_ant_rewards_1000000_2.npy')[1:]

# ppo_timesteps = np.load("ppo_timesteps_mean_1000000.npy")
# ppo_rewards = np.load("ppo_rewards_mean_1000000.npy")
# ppo_timesteps_2 = np.load("ppo_timesteps_1000000_2.npy")
# ppo_rewards_2 = np.load("ppo_rewards_1000000_2.npy")
# ppo_timesteps_3 = np.load("ppo_timesteps_1000000_3.npy")
# ppo_rewards_3 = np.load("ppo_rewards_1000000_3.npy")

def filter_sac(s_timesteps,s_rewards,n=300):
    sac_filtered_timesteps = []
    sac_filtered_rewards = []
    for i, (timestep, reward) in enumerate(zip(s_timesteps, s_rewards)):
        if i==0:
            sac_filtered_timesteps.append(timestep)
            sac_filtered_rewards.append(reward)
        else:
            if reward-sac_filtered_rewards[-1]>-n:
                sac_filtered_timesteps.append(timestep)
                sac_filtered_rewards.append(reward)
    return sac_filtered_timesteps,sac_filtered_rewards

# n=500
# sac_filtered_timesteps,sac_filtered_rewards = filter_sac(sac_timesteps,sac_rewards,n)
# sac_filtered_timesteps_2,sac_filtered_rewards_2 = filter_sac(sac_timesteps_2,sac_rewards_2,n)

# plt.figure(figsize=(10, 6))
# plt.plot(ddpg_timesteps_3, ddpg_rewards_3, label="DDPG", color="blue")
# plt.plot(td3_timesteps_2, td3_rewards_2, label="TD3", color="green")
# plt.plot(sac_filtered_timesteps_2, sac_filtered_rewards_2, label="SAC", color="red")
# plt.plot(ppo_timesteps_2, ppo_rewards_2, label="PPO", color="purple")
# plt.xlabel("Timesteps")
# plt.ylabel("Rewards")
# plt.title("Rewards vs Timesteps for Different Algorithms")
# plt.legend()
# plt.grid()
# plt.show()



common_timesteps = np.linspace(0, 1e6, 500)

# function to interpolate and align data
def interpolate_rewards(timesteps_list, rewards_list, common_x):
    aligned_rewards = []
    for timesteps, rewards in zip(timesteps_list, rewards_list):
        aligned_rewards.append(np.interp(common_x, timesteps, rewards))
    return np.array(aligned_rewards)

# RANDOM
# random_rewards_aligned = interpolate_rewards(
#     [random_timesteps, random_timesteps_2, random_timesteps_3],
#     [random_rewards, random_rewards_2, random_rewards_3],
#     common_timesteps
# )
# random_mean = random_rewards_aligned.mean(axis=0)
# random_std = random_rewards_aligned.std(axis=0)

# # DDPG
# ddpg_rewards_aligned = interpolate_rewards(
#     [ddpg_timesteps, ddpg_timesteps_2, ddpg_timesteps_3],
#     [ddpg_rewards, ddpg_rewards_2, ddpg_rewards_3],
#     common_timesteps
# )
# ddpg_mean = ddpg_rewards_aligned.mean(axis=0)
# ddpg_std = ddpg_rewards_aligned.std(axis=0)

# # TD3
# td3_rewards_aligned = interpolate_rewards(
#     [td3_timesteps, td3_timesteps_2, td3_timesteps_3],
#     [td3_rewards, td3_rewards_2, td3_rewards_3],
#     common_timesteps
# )
# td3_mean = td3_rewards_aligned.mean(axis=0)
# td3_std = td3_rewards_aligned.std(axis=0)

# # SAC
# sac_rewards_aligned = interpolate_rewards(
#     [sac_filtered_timesteps, sac_filtered_timesteps_2],
#     [sac_filtered_rewards, sac_filtered_rewards_2],
#     common_timesteps
# )
# sac_mean = sac_rewards_aligned.mean(axis=0)
# sac_std = sac_rewards_aligned.std(axis=0)

# # PPO
# ppo_rewards_aligned = interpolate_rewards(
#     [ppo_timesteps, ppo_timesteps_2, ppo_timesteps_3],
#     [ppo_rewards, ppo_rewards_2, ppo_rewards_3],
#     common_timesteps
# )
# ppo_mean = ppo_rewards_aligned.mean(axis=0)
# ppo_std = ppo_rewards_aligned.std(axis=0)

# # plotting with mean and std
# plt.figure(figsize=(10, 6))

# plt.plot(common_timesteps, random_mean, label="Random", color="black")
# plt.fill_between(common_timesteps, random_mean - random_std, random_mean + random_std, color="black", alpha=0.2)

# plt.plot(common_timesteps, ddpg_mean, label="DDPG", color="blue")
# plt.fill_between(common_timesteps, ddpg_mean - ddpg_std, ddpg_mean + ddpg_std, color="blue", alpha=0.2)

# plt.plot(common_timesteps, td3_mean, label="TD3", color="green")
# plt.fill_between(common_timesteps, td3_mean - td3_std, td3_mean + td3_std, color="green", alpha=0.2)

# plt.plot(common_timesteps, sac_mean, label="SAC", color="red")
# plt.fill_between(common_timesteps, sac_mean - sac_std, sac_mean + sac_std, color="red", alpha=0.2)

# plt.plot(common_timesteps, ppo_mean, label="PPO", color="purple")
# plt.fill_between(common_timesteps, ppo_mean - ppo_std, ppo_mean + ppo_std, color="purple", alpha=0.2)

# plt.xlabel("Timestep")
# plt.ylabel("Average Reward")
# # plt.title("Rewards vs Timesteps")
# plt.legend()
# plt.grid()
# plt.show()