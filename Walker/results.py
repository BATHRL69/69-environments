import agent, sac, ppo, ddpg, td3
import gymnasium as gym
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = sac.SACAgent(env)
# train_agent.load("sac_ant_1000000.data")
# make_video(env,train_agent,"sac_ant_1000000_vid.mp4")
# SAVE_PATH = "sac_ant_temp.data"
# train_agent = sac.SACAgent(env)
# # train_agent.train(num_timesteps=1_000_000, start_timesteps=10000)
# train_agent.load(SAVE_PATH)

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
