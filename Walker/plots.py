import numpy as np
import matplotlib.pyplot as plt

#LOADING DATA
dpo_timesteps = np.load('dpo_ant_timesteps_1000000_0.npy')
dpo_rewards = np.load('dpo_ant_rewards_1000000_0.npy')
dpo_timesteps_2 = np.load('dpo_ant_timesteps_1000000_1.npy')
dpo_rewards_2 = np.load('dpo_ant_rewards_1000000_1.npy')
dpo_timesteps_3 = np.load('dpo_ant_timesteps_1000000_2.npy')
dpo_rewards_3 = np.load('dpo_ant_rewards_1000000_2.npy')

ppo_timesteps = np.load("ppo_timesteps_mean_1000000.npy")
ppo_rewards = np.load("ppo_rewards_mean_1000000.npy")
ppo_timesteps_2 = np.load("ppo_timesteps_1000000_2.npy")
ppo_rewards_2 = np.load("ppo_rewards_1000000_2.npy")
ppo_timesteps_3 = np.load("ppo_timesteps_1000000_3.npy")
ppo_rewards_3 = np.load("ppo_rewards_1000000_3.npy")

random_timesteps = np.load("random_timesteps_1000000_1.npy")
random_rewards = np.load("random_rewards_1000000_1.npy")
random_timesteps_2 = np.load("random_timesteps_1000000_2.npy")
random_rewards_2 = np.load("random_rewards_1000000_2.npy")
random_timesteps_3 = np.load("random_timesteps_1000000_3.npy")
random_rewards_3 = np.load("random_rewards_1000000_3.npy")

ddpg_timesteps = np.load("new_ddpg_ant_timesteps_1000000_0.npy")
ddpg_rewards = np.load("new_ddpg_ant_rewards_1000000_0.npy")
ddpg_timesteps_2 = np.load("new_ddpg_ant_timesteps_1000000_1.npy")
ddpg_rewards_2 = np.load("new_ddpg_ant_rewards_1000000_1.npy")
ddpg_timesteps_3 = np.load("new_ddpg_ant_timesteps_1000000_2.npy")
ddpg_rewards_3 = np.load("new_ddpg_ant_rewards_1000000_2.npy")

td3_timesteps = np.load("new_td3_ant_timesteps_1000000_0.npy")
td3_rewards = np.load("new_td3_ant_rewards_1000000_0.npy")
td3_timesteps_2 = np.load("new_td3_ant_timesteps_1000000_1.npy")
td3_rewards_2 = np.load("new_td3_ant_rewards_1000000_1.npy")
td3_timesteps_3 = np.load("new_td3_ant_timesteps_1000000_2.npy")
td3_rewards_3 = np.load("new_td3_ant_rewards_1000000_2.npy")

sac_timesteps = np.load('new1_sac_ant_timesteps_1000000_0.npy')
sac_rewards = np.load('new1_sac_ant_rewards_1000000_0.npy')
sac_timesteps_2 = np.load('new1_sac_ant_timesteps_1000000_1.npy')
sac_rewards_2 = np.load('new1_sac_ant_rewards_1000000_1.npy')
sac_timesteps_3 = np.load('new1_sac_ant_timesteps_1000000_2.npy')
sac_rewards_3 = np.load('new1_sac_ant_rewards_1000000_2.npy')

sac_tuned_timesteps = np.load('sac_tuned_ant_timesteps_1000000_0.npy')
sac_tuned_rewards = np.load('sac_tuned_ant_rewards_1000000_0.npy')
sac_tuned_timesteps_2 = np.load('sac_tuned_ant_timesteps_1000000_1.npy')
sac_tuned_rewards_2 = np.load('sac_tuned_ant_rewards_1000000_1.npy')
sac_tuned_timesteps_3 = np.load('sac_tuned_ant_timesteps_1000000_2.npy')
sac_tuned_rewards_3 = np.load('sac_tuned_ant_rewards_1000000_2.npy')

def interpolate_rewards(timesteps_list, rewards_list, common_x):
    aligned_rewards = []
    for timesteps, rewards in zip(timesteps_list, rewards_list):
        aligned_rewards.append(np.interp(common_x, timesteps, rewards))
    return np.array(aligned_rewards)

def moving_average(data, window_size=11):
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='reflect')

    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

def moving_average_thrice(r1,t1,r2,t2,r3,t3,window_size = 25):
  r1 =  moving_average(r1,window_size=window_size)
  t1 = t1[:len(r1)]
  r2 =  moving_average(r2,window_size=window_size)
  t2 = t2[:len(r2)]
  r3 =  moving_average(r3,window_size=window_size)
  t3 = t3[:len(r3)]
  return r1,t1,r2,t2,r3,t3

window_size=25
sac_rewards,sac_timesteps,sac_rewards_2,sac_timesteps_2,sac_rewards_3,sac_timesteps_3 = moving_average_thrice(sac_rewards,sac_timesteps,sac_rewards_2,sac_timesteps_2,sac_rewards_3,sac_timesteps_3,window_size=window_size)

sac_tuned_rewards,sac_tuned_timesteps,sac_tuned_rewards_2,sac_tuned_timesteps_2,sac_tuned_rewards_3,sac_tuned_timesteps_3 = moving_average_thrice(sac_tuned_rewards,sac_tuned_timesteps,sac_tuned_rewards_2,sac_tuned_timesteps_2,sac_tuned_rewards_3,sac_tuned_timesteps_3,window_size=window_size)

random_rewards,random_timesteps,random_rewards_2,random_timesteps_2,random_rewards_3,random_timesteps_3 = moving_average_thrice(random_rewards,random_timesteps,random_rewards_2,random_timesteps_2,random_rewards_3,random_timesteps_3,window_size=window_size)

ppo_rewards,ppo_timesteps,ppo_rewards_2,ppo_timesteps_2,ppo_rewards_3,ppo_timesteps_3 = moving_average_thrice(ppo_rewards,ppo_timesteps,ppo_rewards_2,ppo_timesteps_2,ppo_rewards_3,ppo_timesteps_3,window_size=window_size)

dpo_rewards,dpo_timesteps,dpo_rewards_2,dpo_timesteps_2,dpo_rewards_3,dpo_timesteps_3 = moving_average_thrice(dpo_rewards,dpo_timesteps,dpo_rewards_2,dpo_timesteps_2,dpo_rewards_3,dpo_timesteps_3,window_size=window_size)

ddpg_rewards,ddpg_timesteps,ddpg_rewards_2,ddpg_timesteps_2,ddpg_rewards_3,ddpg_timesteps_3 = moving_average_thrice(ddpg_rewards,ddpg_timesteps,ddpg_rewards_2,ddpg_timesteps_2,ddpg_rewards_3,ddpg_timesteps_3,window_size=window_size)

td3_rewards,td3_timesteps,td3_rewards_2,td3_timesteps_2,td3_rewards_3,td3_timesteps_3 = moving_average_thrice(td3_rewards,td3_timesteps,td3_rewards_2,td3_timesteps_2,td3_rewards_3,td3_timesteps_3,window_size=window_size)

common_timesteps = np.linspace(0, 1e6, 500)

#RANDOM
random_rewards_aligned = interpolate_rewards(
    [random_timesteps, random_timesteps_2, random_timesteps_3],
    [random_rewards, random_rewards_2, random_rewards_3],
    common_timesteps
)
random_mean = random_rewards_aligned.mean(axis=0)
random_std = random_rewards_aligned.std(axis=0)

# DDPG
ddpg_rewards_aligned = interpolate_rewards(
    [ddpg_timesteps, ddpg_timesteps_2, ddpg_timesteps_3],
    [ddpg_rewards, ddpg_rewards_2, ddpg_rewards_3],
    common_timesteps
)
ddpg_mean = ddpg_rewards_aligned.mean(axis=0)
ddpg_std = ddpg_rewards_aligned.std(axis=0)

#TD3
td3_rewards_aligned = interpolate_rewards(
    [td3_timesteps, td3_timesteps_2,td3_timesteps_3],
    [td3_rewards, td3_rewards_2,td3_rewards_3],
    common_timesteps
)
td3_mean = td3_rewards_aligned.mean(axis=0)
td3_std = td3_rewards_aligned.std(axis=0)

#PPO
ppo_rewards_aligned = interpolate_rewards(
    [ppo_timesteps, ppo_timesteps_2, ppo_timesteps_3],
    [ppo_rewards, ppo_rewards_2, ppo_rewards_3],
    common_timesteps
)
ppo_mean = ppo_rewards_aligned.mean(axis=0)
ppo_std = ppo_rewards_aligned.std(axis=0)

#SAC
sac_rewards_aligned = interpolate_rewards(
    [sac_timesteps, sac_timesteps_2,sac_timesteps_3],
    [sac_rewards, sac_rewards_2,sac_rewards_3],
    common_timesteps
)
sac_mean = sac_rewards_aligned.mean(axis=0)
sac_std = sac_rewards_aligned.std(axis=0)

#DPO
dpo_rewards_aligned = interpolate_rewards(
    [dpo_timesteps, dpo_timesteps_2,dpo_timesteps_3],
    [dpo_rewards, dpo_rewards_2,dpo_rewards_3],
    common_timesteps
)

dpo_mean = dpo_rewards_aligned.mean(axis=0)
dpo_std = dpo_rewards_aligned.std(axis=0)

#TUNED SAC
sac_tuned_rewards_aligned = interpolate_rewards(
    [sac_tuned_timesteps, sac_tuned_timesteps_2,sac_tuned_timesteps_3],
    [sac_tuned_rewards, sac_tuned_rewards_2,sac_tuned_rewards_3],
    common_timesteps
)
sac_tuned_mean = sac_tuned_rewards_aligned.mean(axis=0)
sac_tuned_std = sac_tuned_rewards_aligned.std(axis=0)

# plt.figure(figsize=(10, 6))

# plt.plot(common_timesteps, random_mean, label="Random", color="black")
# plt.fill_between(common_timesteps, random_mean - random_std, random_mean + random_std, color="black", alpha=0.2)

# plt.plot(common_timesteps, ddpg_mean, label="DDPG", color="blue")
# plt.fill_between(common_timesteps, ddpg_mean - ddpg_std, ddpg_mean + ddpg_std, color="blue", alpha=0.2)

# plt.plot(common_timesteps, td3_mean, label="TD3", color="green")
# plt.fill_between(common_timesteps, td3_mean - td3_std, td3_mean + td3_std, color="green", alpha=0.2)

# plt.plot(common_timesteps, sac_mean, label="SAC", color="red")
# plt.fill_between(common_timesteps, sac_mean - sac_std, sac_mean + sac_std, color="red", alpha=0.2)

# plt.plot(common_timesteps, sac_tuned_mean, label="SAC (Tuned)", color="orange")
# plt.fill_between(common_timesteps, sac_tuned_mean - sac_tuned_std, sac_tuned_mean + sac_tuned_std, color="orange", alpha=0.2)

# plt.plot(common_timesteps, dpo_mean, label="DPO", color="brown")
# plt.fill_between(common_timesteps, dpo_mean - dpo_std, dpo_mean + dpo_std, color="brown", alpha=0.2)

# plt.plot(common_timesteps, ppo_mean, label="PPO", color="purple")
# plt.fill_between(common_timesteps, ppo_mean - ppo_std, ppo_mean + ppo_std, color="purple", alpha=0.2)

# plt.xlabel("Timesteps")
# plt.ylabel("Average Reward")
# plt.legend()
# plt.grid()
# plt.show()

# Humanoid

humanoid_random_timesteps = np.load("random_humanoid_timesteps_1000000_0.npy")
humanoid_random_rewards = np.load("random_humanoid_rewards_1000000_0.npy")
humanoid_random_timesteps_2 = np.load("random_humanoid_timesteps_1000000_1.npy")
humanoid_random_rewards_2 = np.load("random_humanoid_rewards_1000000_1.npy")
humanoid_random_timesteps_3 = np.load("random_humanoid_timesteps_1000000_2.npy")
humanoid_random_rewards_3 = np.load("random_humanoid_rewards_1000000_2.npy")

humanoid_td3_timesteps = np.load("new_td3_humanoid_timesteps_1000000_0.npy")
humanoid_td3_rewards = np.load("new_td3_humanoid_rewards_1000000_0.npy")
humanoid_td3_timesteps_2 = np.load("new_td3_humanoid_timesteps_1000000_1.npy")
humanoid_td3_rewards_2 = np.load("new_td3_humanoid_rewards_1000000_1.npy")
humanoid_td3_timesteps_3 = np.load("new_td3_humanoid_timesteps_1000000_2.npy")
humanoid_td3_rewards_3 = np.load("new_td3_humanoid_rewards_1000000_2.npy")

humanoid_sac_timesteps = np.load('new_fe_sac_humanoid_timesteps_1000000_0.npy')
humanoid_sac_rewards = np.load('new_fe_sac_humanoid_rewards_1000000_0.npy')
humanoid_sac_timesteps_2 = np.load('new_fe_sac_humanoid_timesteps_1000000_1.npy')
humanoid_sac_rewards_2 = np.load('new_fe_sac_humanoid_rewards_1000000_1.npy')
humanoid_sac_timesteps_3 = np.load('new_fe_sac_humanoid_timesteps_1000000_2.npy')
humanoid_sac_rewards_3 = np.load('new_fe_sac_humanoid_rewards_1000000_2.npy')

humanoid_sac_tuned_timesteps = np.load('new_ae_sac_humanoid_timesteps_1000000_0.npy')
humanoid_sac_tuned_rewards = np.load('new_ae_sac_humanoid_rewards_1000000_0.npy')
humanoid_sac_tuned_timesteps_2 = np.load('new_ae_sac_humanoid_timesteps_1000000_1.npy')
humanoid_sac_tuned_rewards_2 = np.load('new_ae_sac_humanoid_rewards_1000000_1.npy')
humanoid_sac_tuned_timesteps_3 = np.load('new_ae_sac_humanoid_timesteps_1000000_2.npy')
humanoid_sac_tuned_rewards_3 = np.load('new_ae_sac_humanoid_rewards_1000000_2.npy')

def moving_average_thrice(r1,t1,r2,t2,r3,t3,window_size = 25):
  r1 =  moving_average(r1,window_size=window_size)
  t1 = t1[:len(r1)]
  r2 =  moving_average(r2,window_size=window_size)
  t2 = t2[:len(r2)]
  r3 =  moving_average(r3,window_size=window_size)
  t3 = t3[:len(r3)]
  return r1,t1,r2,t2,r3,t3

humanoid_td3_rewards,humanoid_td3_timesteps,humanoid_td3_rewards_2,humanoid_td3_timesteps_2,humanoid_td3_rewards_3,humanoid_td3_timesteps_3 = moving_average_thrice(humanoid_td3_rewards,humanoid_td3_timesteps,humanoid_td3_rewards_2,humanoid_td3_timesteps_2,humanoid_td3_rewards_3,humanoid_td3_timesteps_3)

humanoid_sac_rewards,humanoid_sac_timesteps,humanoid_sac_rewards_2,humanoid_sac_timesteps_2,humanoid_sac_rewards_3,humanoid_sac_timesteps_3 = moving_average_thrice(humanoid_sac_rewards,humanoid_sac_timesteps,humanoid_sac_rewards_2,humanoid_sac_timesteps_2,humanoid_sac_rewards_3,humanoid_sac_timesteps_3)

humanoid_random_rewards,humanoid_random_timesteps,humanoid_random_rewards_2,humanoid_random_timesteps_2,humanoid_random_rewards_3,humanoid_random_timesteps_3 = moving_average_thrice(humanoid_random_rewards,humanoid_random_timesteps,humanoid_random_rewards_2,humanoid_random_timesteps_2,humanoid_random_rewards_3,humanoid_random_timesteps_3)

humanoid_sac_tuned_rewards,humanoid_sac_tuned_timesteps,humanoid_sac_tuned_rewards_2,humanoid_sac_tuned_timesteps_2,humanoid_sac_tuned_rewards_3,humanoid_sac_tuned_timesteps_3 = moving_average_thrice(humanoid_sac_tuned_rewards,humanoid_sac_tuned_timesteps,humanoid_sac_tuned_rewards_2,humanoid_sac_tuned_timesteps_2,humanoid_sac_tuned_rewards_3,humanoid_sac_tuned_timesteps_3)

common_timesteps = np.linspace(0, 1e6, 500)

#RANDOM
humanoid_random_rewards_aligned = interpolate_rewards(
    [humanoid_random_timesteps, humanoid_random_timesteps_2, humanoid_random_timesteps_3],
    [humanoid_random_rewards, humanoid_random_rewards_2, humanoid_random_rewards_3],
    common_timesteps
)
humanoid_random_mean = humanoid_random_rewards_aligned.mean(axis=0)
humanoid_random_std = humanoid_random_rewards_aligned.std(axis=0)

#TD3
humanoid_td3_rewards_aligned = interpolate_rewards(
    [humanoid_td3_timesteps, humanoid_td3_timesteps_2,humanoid_td3_timesteps_3],
    [humanoid_td3_rewards, humanoid_td3_rewards_2,humanoid_td3_rewards_3],
    common_timesteps
)
humanoid_td3_mean = humanoid_td3_rewards_aligned.mean(axis=0)
humanoid_td3_std = humanoid_td3_rewards_aligned.std(axis=0)

#SAC
humanoid_sac_rewards_aligned = interpolate_rewards(
    [humanoid_sac_timesteps, humanoid_sac_timesteps_2,humanoid_sac_timesteps_3],
    [humanoid_sac_rewards, humanoid_sac_rewards_2,humanoid_sac_rewards_3],
    common_timesteps
)
humanoid_sac_mean = humanoid_sac_rewards_aligned.mean(axis=0)
humanoid_sac_std = humanoid_sac_rewards_aligned.std(axis=0)

#SAC TUNED
humanoid_sac_tuned_rewards_aligned = interpolate_rewards(
    [humanoid_sac_tuned_timesteps, humanoid_sac_tuned_timesteps_2,humanoid_sac_tuned_timesteps_3],
    [humanoid_sac_tuned_rewards, humanoid_sac_tuned_rewards_2,humanoid_sac_tuned_rewards_3],
    common_timesteps
)
humanoid_sac_tuned_mean = humanoid_sac_tuned_rewards_aligned.mean(axis=0)
humanoid_sac_tuned_std = humanoid_sac_tuned_rewards_aligned.std(axis=0)

# # plotting with mean and std
# plt.figure(figsize=(10, 6))

# plt.plot(common_timesteps, humanoid_random_mean, label="Random", color="black")
# plt.fill_between(common_timesteps, humanoid_random_mean - humanoid_random_std, humanoid_random_mean + humanoid_random_std, color="black", alpha=0.2)

# plt.plot(common_timesteps, humanoid_td3_mean, label="TD3", color="green")
# plt.fill_between(common_timesteps, humanoid_td3_mean - humanoid_td3_std, humanoid_td3_mean + humanoid_td3_std, color="green", alpha=0.2)

# plt.plot(common_timesteps, humanoid_sac_mean, label="SAC", color="red")
# plt.fill_between(common_timesteps, humanoid_sac_mean - humanoid_sac_std, humanoid_sac_mean + humanoid_sac_std, color="red", alpha=0.2)

# plt.plot(common_timesteps, humanoid_sac_tuned_mean, label="SAC (Tuned)", color="orange")
# plt.fill_between(common_timesteps, humanoid_sac_tuned_mean - humanoid_sac_tuned_std, humanoid_sac_tuned_mean + humanoid_sac_tuned_std, color="orange", alpha=0.2)


# plt.xlabel("Timesteps")
# plt.ylabel("Average Reward")
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(10, 6))

# plt.plot(common_timesteps, random_mean, label="Random", color="black")
# plt.fill_between(common_timesteps, random_mean - random_std, random_mean + random_std, color="black", alpha=0.2)

# plt.plot(common_timesteps, ddpg_mean, label="DDPG", color="blue")
# plt.fill_between(common_timesteps, ddpg_mean - ddpg_std, ddpg_mean + ddpg_std, color="blue", alpha=0.2)

# plt.plot(common_timesteps, td3_mean, label="TD3", color="green")
# plt.fill_between(common_timesteps, td3_mean - td3_std, td3_mean + td3_std, color="green", alpha=0.2)

# plt.plot(common_timesteps, sac_mean, label="SAC", color="red")
# plt.fill_between(common_timesteps, sac_mean - sac_std, sac_mean + sac_std, color="red", alpha=0.2)

# plt.plot(common_timesteps, sac_tuned_mean, label="SAC (Tuned)", color="orange")
# plt.fill_between(common_timesteps, sac_tuned_mean - sac_tuned_std, sac_tuned_mean + sac_tuned_std, color="orange", alpha=0.2)

# plt.plot(common_timesteps, dpo_mean, label="DPO", color="brown")
# plt.fill_between(common_timesteps, dpo_mean - dpo_std, dpo_mean + dpo_std, color="brown", alpha=0.2)

# plt.plot(common_timesteps, ppo_mean, label="PPO", color="purple")
# plt.fill_between(common_timesteps, ppo_mean - ppo_std, ppo_mean + ppo_std, color="purple", alpha=0.2)

# plt.xlabel("Timesteps")
# plt.ylabel("Average Reward")
# plt.legend()
# plt.grid()
# plt.show()
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

axs[0].plot(common_timesteps, random_mean, label="Random", color="black")
axs[0].fill_between(
    common_timesteps,
    random_mean - random_std,
    random_mean + random_std,
    color="black",
    alpha=0.2,
)
axs[0].plot(common_timesteps, ddpg_mean, label="DDPG", color="blue")
axs[0].fill_between(
    common_timesteps,
    ddpg_mean - ddpg_std,
    ddpg_mean + ddpg_std,
    color="blue",
    alpha=0.2,
)
axs[0].plot(common_timesteps, td3_mean, label="TD3", color="green")
axs[0].fill_between(
    common_timesteps,
    td3_mean - td3_std,
    td3_mean + td3_std,
    color="green",
    alpha=0.2,
)
axs[0].plot(common_timesteps, sac_mean, label="SAC", color="red")
axs[0].fill_between(
    common_timesteps,
    sac_mean - sac_std,
    sac_mean + sac_std,
    color="red",
    alpha=0.2,
)
axs[0].plot(common_timesteps, sac_tuned_mean, label="SAC (Tuned)", color="purple")
axs[0].fill_between(
    common_timesteps,
    sac_tuned_mean - sac_tuned_std,
    sac_tuned_mean + sac_tuned_std,
    color="purple",
    alpha=0.2,
)
axs[0].plot(common_timesteps, dpo_mean, label="DPO", color="orange")
axs[0].fill_between(
    common_timesteps,
    dpo_mean - dpo_std,
    dpo_mean + dpo_std,
    color="orange",
    alpha=0.2,
)
axs[0].plot(common_timesteps, ppo_mean, label="PPO", color="magenta")
axs[0].fill_between(
    common_timesteps,
    ppo_mean - ppo_std,
    ppo_mean + ppo_std,
    color="magenta",
    alpha=0.2,
)
axs[0].set_xlabel("Timesteps")
axs[0].set_ylabel("Average Reward")
axs[0].legend(fontsize=16)
axs[0].grid()
axs[0].set_title("Ant-v4")

axs[1].plot(common_timesteps, humanoid_random_mean, label="Random", color="black")
axs[1].fill_between(
    common_timesteps,
    humanoid_random_mean - humanoid_random_std,
    humanoid_random_mean + humanoid_random_std,
    color="black",
    alpha=0.2,
)
axs[1].plot(common_timesteps, humanoid_td3_mean, label="TD3", color="green")
axs[1].fill_between(
    common_timesteps,
    humanoid_td3_mean - humanoid_td3_std,
    humanoid_td3_mean + humanoid_td3_std,
    color="green",
    alpha=0.2,
)
axs[1].plot(common_timesteps, humanoid_sac_mean, label="SAC", color="red")
axs[1].fill_between(
    common_timesteps,
    humanoid_sac_mean - humanoid_sac_std,
    humanoid_sac_mean + humanoid_sac_std,
    color="red",
    alpha=0.2,
)
axs[1].plot(common_timesteps, humanoid_sac_tuned_mean, label="SAC (Tuned)", color="purple")
axs[1].fill_between(
    common_timesteps,
    humanoid_sac_tuned_mean - humanoid_sac_tuned_std,
    humanoid_sac_tuned_mean + humanoid_sac_tuned_std,
    color="purple",
    alpha=0.2,
)
axs[1].set_xlabel("Timesteps")
axs[1].legend(fontsize=16)
axs[1].grid()
axs[1].set_title("Humanoid-v4")

plt.tight_layout()
plt.show()

sac_max_reward_loc = np.argmax(sac_mean)
sac_max_mean_reward = sac_mean[sac_max_reward_loc]
sac_max_std_reward = sac_std[sac_max_reward_loc]

ppo_max_reward_loc = np.argmax(ppo_mean)
ppo_max_mean_reward = ppo_mean[ppo_max_reward_loc]
ppo_max_std_reward = ppo_std[ppo_max_reward_loc]

dpo_max_reward_loc = np.argmax(dpo_mean)
dpo_max_mean_reward = dpo_mean[dpo_max_reward_loc]
dpo_max_std_reward = dpo_std[dpo_max_reward_loc]

ddpg_max_reward_loc = np.argmax(ddpg_mean)
ddpg_max_mean_reward = ddpg_mean[ddpg_max_reward_loc]
ddpg_max_std_reward = ddpg_std[ddpg_max_reward_loc]

td3_max_reward_loc = np.argmax(td3_mean)
td3_max_mean_reward = td3_mean[td3_max_reward_loc]
td3_max_std_reward = td3_std[td3_max_reward_loc]

sac_tuned_max_reward_loc = np.argmax(sac_tuned_mean)
sac_tuned_max_mean_reward = sac_tuned_mean[sac_tuned_max_reward_loc]
sac_tuned_max_std_reward = sac_tuned_std[sac_tuned_max_reward_loc]

random_max_reward_loc = np.argmax(random_mean)
random_max_mean_reward = random_mean[random_max_reward_loc]
random_max_std_reward = random_std[random_max_reward_loc]

import pandas as pd

methods = ["SAC", "PPO", "DPO", "DDPG", "TD3", "SAC (Tuned)","Random"]
data = [
    (sac_max_mean_reward, sac_max_std_reward),
    (ppo_max_mean_reward, ppo_max_std_reward),
    (dpo_max_mean_reward, dpo_max_std_reward),
    (ddpg_max_mean_reward, ddpg_max_std_reward),
    (td3_max_mean_reward, td3_max_std_reward),
    (sac_tuned_max_mean_reward, sac_tuned_max_std_reward),
    (random_max_mean_reward,random_max_std_reward)
]

data = [(round(a,2),round(b,2)) for a,b in data]

table = pd.DataFrame(data, columns=["Max Mean Reward", "Max Std Reward"], index=methods)

table["Mean ± Std"] = table["Max Mean Reward"].astype(str) + " ± " + table["Max Std Reward"].astype(str)

table = table[["Mean ± Std"]]

print(table)

sac_max_reward_loc = np.argmax(humanoid_sac_mean)
sac_max_mean_reward = humanoid_sac_mean[sac_max_reward_loc]
sac_max_std_reward = humanoid_sac_std[sac_max_reward_loc]

td3_max_reward_loc = np.argmax(humanoid_td3_mean)
td3_max_mean_reward = humanoid_td3_mean[td3_max_reward_loc]
td3_max_std_reward = humanoid_td3_std[td3_max_reward_loc]

sac_tuned_max_reward_loc = np.argmax(humanoid_sac_tuned_mean)
sac_tuned_max_mean_reward = humanoid_sac_tuned_mean[sac_tuned_max_reward_loc]
sac_tuned_max_std_reward = humanoid_sac_tuned_std[sac_tuned_max_reward_loc]

random_max_reward_loc = np.argmax(humanoid_random_mean)
random_max_mean_reward = humanoid_random_mean[random_max_reward_loc]
random_max_std_reward = humanoid_random_std[random_max_reward_loc]

methods = ["SAC", "TD3", "SAC (Tuned)","Random"]
data = [
    (sac_max_mean_reward, sac_max_std_reward),
    (td3_max_mean_reward, td3_max_std_reward),
    (sac_tuned_max_mean_reward, sac_tuned_max_std_reward),
    (random_max_mean_reward,random_max_std_reward)
]

data = [(round(a,2),round(b,2)) for a,b in data]

table = pd.DataFrame(data, columns=["Max Mean Reward", "Max Std Reward"], index=methods)

table["Mean ± Std"] = table["Max Mean Reward"].astype(str) + " ± " + table["Max Std Reward"].astype(str)

table = table[["Mean ± Std"]]

print(table)