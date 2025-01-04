import agent, sac, ppo, ddpg, td3
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

#### SAC AGENT

# train_agent = sac.SACAgent(env)
# train_agent.train(num_timesteps=1_000_00, start_timesteps=1_000_00)
# np.save("temp.npy", np.array(train_agent.timestep_list))
# np.save("temp.npy", np.array(train_agent.reward_list))

#### DDPG AGENT

# train_agent = ddpg.DDPGAgent(env,num_train_episodes=1_000_0,actor_lr = 0.0003,critic_lr=0.0003,training_frequency=1,replay_sample_size=256,max_buffer_size=1000000)
# train_agent.train(1_000_000)
# timestep_list = ddpg.GLOBAL_TIMESTEPS
# reward_list = ddpg.GLOBAL_REWARDS
# np.save("new_ddpg_ant_timesteps_1000000.npy", np.array(timestep_list))
# np.save("new_ddpg_ant_rewards_1000000.npy", np.array(reward_list))

### TD3 AGENT
# train_agent = td3.TD3Agent(env,num_train_episodes=1_000_0,actor_lr=0.0003,critic_lr=0.0003,training_frequency=1,actor_update_frequency=2,replay_sample_size=256,max_buffer_size=1000000)
# train_agent.train(1_000_000)
# timestep_list = ddpg.GLOBAL_TIMESTEPS
# reward_list = ddpg.GLOBAL_REWARDS
# np.save("new_ddpg_ant_timesteps_1000000.npy", np.array(timestep_list))
# np.save("new_ddpg_ant_rewards_1000000.npy", np.array(reward_list))

### PPO AGENT

env = gym.make("Ant-v4", render_mode="rgb_array")
train_agent = ppo.PPOAgent(env, observation_space=27, action_space=8, std=0.6)
timestep_list_ppo, reward_list_ppo = train_agent.efficient_train(1_000_000)
np.save("ppo_timesteps_1000000.npy", np.array(timestep_list_ppo))
np.save("ppo_rewards_1000000.npy", np.array(reward_list_ppo))
# env.close()

### DPO AGENT

# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = ppo.DPOAgent(env, observation_space=27, action_space=8, std=0.6)
# timestep_list_ppo, reward_list_ppo = train_agent.efficient_train(1_000_000)
# np.save("dpo_timesteps_1000000.npy", np.array(timestep_list_ppo))
# np.save("dpo_rewards_1000000.npy", np.array(reward_list_ppo))
# env.close()
