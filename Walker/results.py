import agent, sac, ppo, ddpg, td3
import gymnasium as gym
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def plot(timesteps,rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, label="reward", color="red")
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Timesteps")
    plt.legend()
    plt.grid()
    plt.show()

def make_video_sample(env_name,agent,save_path):
    video_env = gym.make(env_name,render_mode="rgb_array")
    frames = []
    state, _ = video_env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = video_env.render()
        frames.append(frame)

        action = agent.actor.sample(torch.Tensor([state]))
        state, reward, done, truncated ,info = video_env.step(action[0].detach().numpy()[0])

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    video_env.close()

def make_video_predict(env,agent,save_path):
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        action = agent.predict(torch.Tensor([state]))
        state, reward, done, truncated ,info = env.step(action[0])

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

def make_video_ddpg(env_name,agent:ddpg.DDPGAgent,save_path):
    video_env = gym.make(env_name,render_mode="rgb_array")
    frames = []
    state, _ = video_env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = video_env.render()
        frames.append(frame)

        # action = agent.predict(torch.Tensor(state))
        # state, reward, done, truncated, info = env.step(action)
        action = agent.actor.get_action(torch.Tensor([state]),test=False)
        state, reward, done, truncated ,info = video_env.step(action[0].detach().numpy())

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    video_env.close()

# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = sac.SACAgent(env)
# train_agent.load("sac_ant_1000000.data")
# make_video(env,train_agent,"sac_ant_1000000_vid.mp4")
# SAVE_PATH = "sac_ant_temp.data"
# train_agent = sac.SACAgent(env)
# # train_agent.train(num_timesteps=1_000_000, start_timesteps=10000)
# train_agent.load(SAVE_PATH)


# np.save("sac_ant_brr_timesteps.npy",np.array(train_agent.timestep_list))
# np.save("sac_ant_brr_rewards.npy",np.array(train_agent.reward_list))

# SAVE_PATH = "temp.data"
# train_agent = sac.SACAgent(env)
# train_agent.train(num_timesteps=1_000_00, start_timesteps=1_000_00)
# train_agent.save(SAVE_PATH)
# train_agent.load(SAVE_PATH)
# train_agent.render()
# make_video_sample(env,train_agent,"sgrgr.mp4")
# np.save("temp.npy", np.array(train_agent.timestep_list))
# np.save("temp.npy", np.array(train_agent.reward_list))

# DDPG
# for i in range(3):
#     env = gym.make("Ant-v4", render_mode="rgb_array")
#     train_agent = ddpg.DDPGAgent(env,num_train_episodes=1_000_000,actor_lr = 0.0003,critic_lr=0.0003,training_frequency=1,replay_sample_size=256,max_buffer_size=1000000,make_video=True)
#     train_agent.train(25000)
#     timestep_list = ddpg.GLOBAL_TIMESTEPS
#     reward_list = ddpg.GLOBAL_REWARDS
#     make_video_ddpg("Ant-v4",train_agent,"new_ddpg_ant_1000000_"+str(i)+".mp4")
#     np.save("new_ddpg_ant_timesteps_1000000_"+str(i)+".npy", np.array(timestep_list))
#     np.save("new_ddpg_ant_rewards_1000000_"+str(i)+".npy", np.array(reward_list))
#     ddpg.GLOBAL_TIMESTEPS = []
#     ddpg.GLOBAL_REWARDS = []
#     env.close()

# # TD3
# for i in range(1,3):
#     make_video = True if i == 0 else False
#     env = gym.make("Humanoid-v4", render_mode="rgb_array")
#     train_agent = td3.TD3Agent(env,make_video=make_video,num_train_episodes=1_000_000,actor_lr=0.0003,critic_lr=0.0003,training_frequency=1,actor_update_frequency=2,replay_sample_size=256,max_buffer_size=1000000)
#     train_agent.train(start_steps = 25000)
#     timestep_list = td3.GLOBAL_TIMESTEPS
#     reward_list = td3.GLOBAL_REWARDS
#     np.save("td3_hl_humanoid_timesteps_1000000_"+str(i)+".npy", np.array(timestep_list))
#     np.save("td3_hl_humanoid_rewards_1000000_"+str(i)+".npy", np.array(reward_list))
#     make_video_ddpg("Humanoid-v4",train_agent,"td3_hl_humanoid_1000000_"+str(i)+".mp4")
#     td3.GLOBAL_TIMESTEPS = []
#     td3.GLOBAL_REWARDS = []
#     env.close()

# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = sac.SACAgent(env,reward_scale=5,make_video=False)
# train_agent.train(num_timesteps=1_025_000, start_timesteps=25000)
# make_video_sample("Ant-v4",train_agent,"new_sac_ant_1000000_2.mp4")
# np.save("new_sac_ant_timesteps_1000000_2.npy", np.array(train_agent.timestep_list))
# np.save("new_sac_ant_rewards_1000000_2.npy", np.array(train_agent.reward_list))
# env.close()

# PPO
# env = gym.make("Ant-v4", render_mode = "rgb_array")
# train_agent = ppo.PPOAgent(env, observation_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
# train_agent.efficient_train(200_000)
# timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
# reward_list_ppo = ppo.GLOBAL_REWARDS
# # np.save("ppo_humanoid_timesteps_1000000.npy", np.array(timestep_list_ppo))
# # np.save("ppo_humanoid_rewards_1000000.npy", np.array(reward_list_ppo))
# make_video_predict(env,train_agent,"ppo_ant_200000.mp4")
# env.close()

# train_agent = ppo.PPOAgent(env, observation_space=27, action_space=8, std=0.6)
# train_agent.efficient_train(100_000)
# timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
# reward_list_ppo = ppo.GLOBAL_REWARDS
# train_agent.max_std = 0.01
# train_agent.render()
# env.close()
# np.save("ppo_timesteps_mean.npy", np.array(train_agent.timestep_list))
# np.save("ppo_rewards_mean.npy", np.array(train_agent.reward_list))

# # DPO
# for train_steps in [100_000,200_000,500_000]:
#     env = gym.make("Ant-v4", render_mode = "rgb_array")
#     train_agent = ppo.DPOAgent(env, observation_space=27, action_space=8, std=0.6)
#     train_agent.efficient_train(train_steps)
#     timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
#     reward_list_ppo = ppo.GLOBAL_REWARDS
#     # np.save("dpo_ant_timesteps_1000000_.npy", np.array(timestep_list_ppo))
#     # np.save("dpo_ant_rewards_1000000_.npy", np.array(reward_list_ppo))
#     make_video_predict(env,train_agent,"dpo_ant_"+str(train_steps)+".mp4")
#     timestep_list_ppo = [] 
#     reward_list_ppo = []

# for i in range(3):
    # random_timesteps = []
    # random_rewards = []
    # env = gym.make("Humanoid-v4", render_mode = None)
    # state, _ = env.reset()
    # episodic_reward = 0
    # for timestep in range(1_000_000):
    #     action = env.action_space.sample()
    #     new_state, reward, is_finished, is_truncated, _ = env.step(action)
    #     episodic_reward+=reward
    #     if is_finished or is_truncated:
    #         random_timesteps.append(timestep)
    #         random_rewards.append(episodic_reward)
    #         print(f"Timestep: {timestep} Reward:{episodic_reward}")
    #         state, _ = env.reset()
    #         episodic_reward=0
    # np.save("random_humanoid_timesteps_1000000_"+str(i)+".npy",np.array(random_timesteps))
    # np.save("random_humanoid_rewards_1000000_"+str(i)+".npy",np.array(random_rewards))


# our_timesteps = np.load("sac_test_ours_timesteps.npy")
# our_rewards = np.load("sac_test_ours_rewards.npy")
# their_timesteps = np.load("sac_test_theirs_timesteps.npy")
# their_rewards = np.load("sac_test_theirs_rewards.npy")

sac_tuned_timesteps = np.load('sac_tuned_ant_timesteps_1000000_0.npy')
sac_tuned_rewards = np.load('sac_tuned_ant_rewards_1000000_0.npy')
sac_tuned_timesteps_2 = np.load('sac_tuned_ant_timesteps_1000000_1.npy')
sac_tuned_rewards_2 = np.load('sac_tuned_ant_rewards_1000000_1.npy')
sac_tuned_timesteps_3 = np.load('sac_tuned_ant_timesteps_1000000_2.npy')
sac_tuned_rewards_3 = np.load('sac_tuned_ant_rewards_1000000_2.npy')

sac_timesteps = np.load('new1_sac_ant_timesteps_1000000_0.npy')
sac_rewards = np.load('new1_sac_ant_rewards_1000000_0.npy')
sac_timesteps_2 = np.load('new1_sac_ant_timesteps_1000000_1.npy')
sac_rewards_2 = np.load('new1_sac_ant_rewards_1000000_1.npy')
sac_timesteps_3 = np.load('new1_sac_ant_timesteps_1000000_2.npy')
sac_rewards_3 = np.load('new1_sac_ant_rewards_1000000_2.npy')

new_our_timesteps_tuned = np.load("sac_test_ours_tuned_timesteps.npy")
new_our_rewards_tuned = np.load("sac_test_ours_tuned_rewards.npy")

new_our_timesteps_not_tuned = np.load("sac_test_ours_nottuned_timesteps.npy")
new_our_rewards_not_tuned = np.load("sac_test_ours_nottuned_rewards.npy")

new_timesteps_not_tuned = np.load("sac_og_v1_timesteps.npy")
new_rewards_not_tuned = np.load("sac_og_v1_rewards.npy")


def moving_average(data, window_size=11):
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='reflect')

    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

sac_rewards =  moving_average(sac_rewards)
sac_timesteps = sac_timesteps[:len(sac_rewards)]

sac_rewards_2 =  moving_average(sac_rewards_2)
sac_timesteps_2 = sac_timesteps_2[:len(sac_rewards_2)]

sac_rewards_3 =  moving_average(sac_rewards_3)
sac_timesteps_3 = sac_timesteps_3[:len(sac_rewards_3)]

sac_tuned_rewards =  moving_average(sac_tuned_rewards)
sac_tuned_timesteps = sac_tuned_timesteps[:len(sac_tuned_rewards)]

sac_tuned_rewards_2 =  moving_average(sac_tuned_rewards_2)
sac_tuned_timesteps_2 = sac_tuned_timesteps_2[:len(sac_tuned_rewards_2)]

sac_tuned_rewards_3 =  moving_average(sac_tuned_rewards_3)
sac_tuned_timesteps_3 = sac_tuned_timesteps_3[:len(sac_tuned_rewards_3)]

new_our_rewards_tuned =  moving_average(new_our_rewards_tuned)
new_our_timesteps_tuned = new_our_timesteps_tuned[:len(new_our_rewards_tuned)]

new_our_rewards_not_tuned =  moving_average(new_our_rewards_not_tuned)
new_our_timesteps_not_tuned = new_our_timesteps_not_tuned[:len(new_our_rewards_not_tuned)]

new_rewards_not_tuned =  moving_average(new_rewards_not_tuned)
new_timesteps_not_tuned = new_timesteps_not_tuned[:len(new_rewards_not_tuned)]

plt.figure(figsize=(10, 6))
plt.plot(sac_tuned_timesteps, sac_tuned_rewards, label="tuned", color="red")
plt.plot(sac_tuned_timesteps_2, sac_tuned_rewards_2, label="tuned", color="red")
plt.plot(sac_tuned_timesteps_3, sac_tuned_rewards_3, label="tuned", color="red")

plt.plot(sac_timesteps, sac_rewards, label="not", color="green")
plt.plot(sac_timesteps_2, sac_rewards_2, label="not", color="green")
plt.plot(sac_timesteps_3, sac_rewards_3, label="not", color="green")

plt.plot(new_our_timesteps_tuned, new_our_rewards_tuned, label="ours tuned", color="yellow")
plt.plot(new_our_timesteps_not_tuned, new_our_rewards_not_tuned, label="ours not", color="purple")

plt.plot(new_timesteps_not_tuned, new_rewards_not_tuned, label="NEW", color="black")




plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.title("Rewards vs Timesteps")
plt.legend()
plt.grid()
plt.show()
