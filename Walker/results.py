import agent, sac, ppo, ddpg
import gymnasium as gym
import numpy as np
import cv2
import torch

def make_video(env,agent,save_path):
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        action = agent.predict(torch.Tensor(state))
        state, reward, done, truncated, info = env.step(action)

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

env = gym.make("Ant-v4", render_mode="rgb_array")

# SAVE_PATH = "sac_humanoid_1000000.data"
# train_agent = sac.SACAgent(env)
# train_agent.load("sac_humanoid.data")
# train_agent.train(num_timesteps=1_000_000, start_timesteps=10000)

# train_agent.render()

# obs, info = env.reset()
# for i in range(10_000):
#     # THIS DOESNT WORK
#     action = train_agent.predict(torch.Tensor([obs]))[0]
#     obs, reward, done, trunacted ,info = env.step(action)
#     # THIS WORKS
#     # action = train_agent.actor.sample(torch.Tensor([obs]))
#     # obs, reward, done, trunacted ,info = env.step(action[0].detach().numpy()[0])
#     img = env.render()
    
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Double Inverted Pendulum", img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     if done:
#         obs, info = env.reset()

# np.save("sac_timesteps_2000000_temp.npy", np.array(train_agent.timestep_list))
# np.save("sac_rewards_2000000_temp.npy", np.array(train_agent.reward_list))

# DDPG
# train_agent = ddpg.DDPGAgent(env)
# train_agent.train(10000)
# env.close()
# timestep_list = ddpg.GLOBAL_TIMESTEPS
# reward_list = ddpg.GLOBAL_REWARDS

# PPO
<<<<<<< HEAD
# train_agent = ppo.PPOAgent(env, observation_space=27, action_space=8, std=0.3)
# train_agent.efficient_train(1_000_000)
# timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
# reward_list_ppo = ppo.GLOBAL_REWARDS
# make_video(env,train_agent,"ppo_1000000_temp.mp4")
# env.close()

# np.save("ppo_timesteps_mean_1000000.npy", np.array(train_agent.timestep_list))
# np.save("ppo_rewards_mean_1000000.npy", np.array(train_agent.reward_list))
=======
train_agent = ppo.PPOAgent(env, observation_space=27, action_space=8, std=0.6)
train_agent.efficient_train(100_000)
timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
reward_list_ppo = ppo.GLOBAL_REWARDS
train_agent.max_std = 0.01
train_agent.render()
env.close()
np.save("ppo_timesteps_mean.npy", np.array(train_agent.timestep_list))
np.save("ppo_rewards_mean.npy", np.array(train_agent.reward_list))


# DPO
ppo.GLOBAL_TIMESTEPS = []
ppo.GLOBAL_REWARDS = []
train_agent = ppo.DPOAgent(env, observation_space=27, action_space=8, std=0.6)
train_agent.efficient_train(100_000)
timestep_list_ppo = ppo.GLOBAL_TIMESTEPS
reward_list_ppo = ppo.GLOBAL_REWARDS
train_agent.max_std = 0.01
train_agent.render()
env.close()
np.save("dpo_timesteps_mean.npy", np.array(train_agent.timestep_list))
np.save("dpo_rewards_mean.npy", np.array(train_agent.reward_list))
>>>>>>> 85780259279f12056189f9ed9615316cd90704ca
