import agent,sac,ppo,ddpg
import gymnasium as gym
import numpy as np
import cv2
import torch

def make_video(env,agent,save_video_path):
    frames = []
    state, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        action = agent.predict(torch.Tensor([state]))[0]
        state, reward, done, truncated, info = env.step(action)

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_video_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    env.close()

# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = sac.SACAgent(env)
# train_agent.load("sac_ant_1000000.data")
# make_video(env,train_agent,"sac_ant_1000000_vid.mp4")


# SAVE_PATH = "sac_ant_temp.data"
# train_agent = sac.SACAgent(env)
# # train_agent.train(num_timesteps=1_000_000, start_timesteps=10000)
# train_agent.load(SAVE_PATH)

# obs, info = env.reset()

# for i in range(10_000):
#     # action = train_agent.actor.sample(torch.Tensor([obs]))
#     # obs, reward, done, trunacted ,info = env.step(action[0].detach().numpy()[0])
#     action = train_agent.predict(torch.Tensor([obs]))
#     obs, reward, done, trunacted ,info = env.step(action[0])
#     img = env.render()
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Double Inverted Pendulum", img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     if done:
#         obs, info = env.reset()

# env.close()

# np.save("sac_ant_brr_timesteps.npy",np.array(train_agent.timestep_list))
# np.save("sac_ant_brr_rewards.npy",np.array(train_agent.reward_list))
# np.save("sac_ant_brr_mean_rewards.npy",np.array(train_agent.mean_reward_list))



#DDPG
env = gym.make("Ant-v4", render_mode="rgb_array")
train_agent = ddpg.DDPGAgent(env)
train_agent.train(100_000)
timestep_list = ddpg.GLOBAL_TIMESTEPS
reward_list = ddpg.GLOBAL_REWARDS
# train_agent.render()
make_video(env,train_agent,"ddpg_100000.mp4")
env.close()
# np.save("ddpg_ant_100000_timesteps.npy",np.array(timestep_list))
# np.save("ddpg_ant_100000_rewards.npy",np.array(reward_list))


#PPO
# train_agent = ppo.PPOAgent(env,observation_space=27,action_space=8)
# train_agent.train()
# # timestep_list = ppo.GLOBAL_TIMESTEPS
# # reward_list = ppo.GLOBAL_REWARDS
# env.close()
