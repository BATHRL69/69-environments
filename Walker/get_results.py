import sac, ppo, ddpg, td3
import gymnasium as gym
import numpy as np

START_STEPS = 25000
TIMESTEPS = 1000000

# # SAC AGENT
# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = sac.SACAgent(env, fixed_alpha=0.2)
# train_agent.train(num_timesteps=TIMESTEPS, start_timesteps=START_STEPS)
# np.save("sac_timesteps.npy", np.array(train_agent.timestep_list))
# np.save("sac_rewards.npy", np.array(train_agent.reward_list))
# env.close()

# # SAC (TUNED) AGENT
# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = sac.SACAgent(env)
# train_agent.train(num_timesteps=TIMESTEPS, start_timesteps=START_STEPS)
# np.save("sac_tuned_timesteps.npy", np.array(train_agent.timestep_list))
# np.save("sac_tuned_rewards.npy", np.array(train_agent.reward_list))
# env.close()

# # DDPG AGENT
# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = ddpg.DDPGAgent(env)
# train_agent.train(num_timesteps=TIMESTEPS, start_timesteps=START_STEPS)
# np.save("ddpg_timesteps.npy", np.array(train_agent.timestep_list))
# np.save("ddpg_rewards.npy", np.array(train_agent.reward_list))
# env.close()

# # TD3 AGENT
# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = td3.TD3Agent(env)
# train_agent.train(num_timesteps=TIMESTEPS, start_timesteps=START_STEPS)
# np.save("td3_timesteps.npy", np.array(train_agent.timestep_list))
# np.save("td3_rewards.npy", np.array(train_agent.reward_list))
# env.close()

# # PPO AGENT
# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = ppo.PPOAgent(env, observation_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
# timestep_list_ppo, reward_list_ppo = train_agent.efficient_train(num_iterations=TIMESTEPS)
# np.save("ppo_timesteps.npy", np.array(timestep_list_ppo))
# np.save("ppo_rewards.npy", np.array(reward_list_ppo))
# env.close()

# # DPO AGENT
# env = gym.make("Ant-v4", render_mode="rgb_array")
# train_agent = ppo.DPOAgent(env, observation_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
# timestep_list_dpo, reward_list_dpo = train_agent.efficient_train(num_iterations=TIMESTEPS)
# np.save("dpo_timesteps.npy", np.array(timestep_list_dpo))
# np.save("dpo_rewards.npy", np.array(reward_list_dpo))
# env.close()
