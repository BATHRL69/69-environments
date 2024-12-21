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

def make_video_sample(env,agent,save_path):
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        action = agent.actor.sample(torch.Tensor([state]))
        state, reward, done, truncated ,info = env.step(action[0].detach().numpy()[0])

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

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

def make_video_ddpg(env,agent:ddpg.DDPGAgent,save_path):
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        # action = agent.predict(torch.Tensor(state))
        # state, reward, done, truncated, info = env.step(action)
        action = agent.actor.get_action(torch.Tensor([state]),test=False)
        state, reward, done, truncated ,info = env.step(action[0].detach().numpy())

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

env = gym.make("Ant-v4", render_mode="rgb_array")

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
# train_agent = ddpg.DDPGAgent(env,num_train_episodes=1_000_000)
# train_agent.train(10000)
# env.close()
# timestep_list = ddpg.GLOBAL_TIMESTEPS
# reward_list = ddpg.GLOBAL_REWARDS
# make_video_ddpg(env,train_agent,"ddpg_ant_1000000_3.mp4")
# np.save("ddpg_ant_timesteps_1000000_3.npy", np.array(timestep_list))
# np.save("ddpg_ant_rewards_1000000_3.npy", np.array(reward_list))

#TD3
# train_agent = td3.TD3Agent(env,num_train_episodes=1_000_000,actor_lr=0.0003,critic_lr=0.0003,training_frequency=1,actor_update_frequency=1)
# train_agent.train(10000)
# timestep_list = td3.GLOBAL_TIMESTEPS
# reward_list = td3.GLOBAL_REWARDS
# make_video_ddpg(env,train_agent,"td3_ant_1000000_newlr.mp4")
# plot(timestep_list,reward_list)
# np.save("td3_ant_timesteps_1000000_newlr.npy", np.array(timestep_list))
# np.save("td3_ant_rewards_1000000_newlr.npy", np.array(reward_list))

# PPO
# train_agent = ppo.PPOAgent(env, observation_space=27, action_space=8, std=0.3)
# train_agent.efficient_train(1_000_000)
# timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
# reward_list_ppo = ppo.GLOBAL_REWARDS
# np.save("ppo_timesteps_1000000_3.npy", np.array(timestep_list_ppo))
# np.save("ppo_rewards_1000000_3.npy", np.array(reward_list_ppo))
# # make_video_predict(env,train_agent,"ppo_1000000_vid_2.mp4")
# # env.close()

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
# train_agent = ppo.DPOAgent(env, observation_space=27, action_space=8, std=0.6)
# train_agent.efficient_train(1_000_00)
# timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
# reward_list_ppo = ppo.GLOBAL_REWARDS
# train_agent.max_std = 0.01
# train_agent.render()
# make_video_predict(env,train_agent,"dpo_ant_1000000.mp4")
# np.save("dpo_ant_timesteps_1000000.npy", np.array(timestep_list_ppo))
# np.save("dpo_ant_rewards_1000000.npy", np.array(timestep_list_ppo))
