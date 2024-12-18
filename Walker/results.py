import agent, sac, ppo, ddpg
import gymnasium as gym
import numpy as np

env = gym.make("Ant-v4", render_mode="rgb_array")

SAVE_PATH = "sac_ant_500000.data"
train_agent = sac.SACAgent(env)
train_agent.train(num_timesteps=500_000, start_timesteps=10000)
train_agent.save(SAVE_PATH)
# train_agent.render()
env.close()

np.save("sac_timesteps_mean_500000.npy", np.array(train_agent.timestep_list))
np.save("sac_rewards_mean_500000.npy", np.array(train_agent.reward_list))

# DDPG
# train_agent = ddpg.DDPGAgent(env)
# train_agent.train(10000)
# env.close()
# timestep_list = ddpg.GLOBAL_TIMESTEPS
# reward_list = ddpg.GLOBAL_REWARDS

# PPO
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
