import gymnasium as gym
import numpy as np
import os
import pickle
import random
import torch

from agent import Agent
from itertools import product

# class SarsaAgent(Agent):

#   def __init__(self, env: gym.Env, n_actions=50, alpha=0.1, gamma=0.99, epsilon=0.1):
#     self.env = env
#     self.n_actions = n_actions
#     self.alpha = alpha
#     self.gamma = gamma
#     self.epsilon = epsilon
#     self.q_table = {}
  

#   def get_actions(self):
#     return [i for i in range(self.n_actions)]


#   def get_continuous_action(self, action):
#     # hard coding for now using (-3, 3) bounds on pendulum environment
#     return (6 * (action + 0.5) / self.n_actions) - 3


#   def choose_action(self, state, epsilon_override=None):
#     epsilon = self.epsilon if epsilon_override is None else epsilon_override
#     actions = self.get_actions()

#     if state not in self.q_table:
#         self.q_table[state] = {a : np.random.rand() for a in actions}

#     return random.choice(actions) if np.random.rand() < epsilon else max(self.q_table[state], key=self.q_table[state].get)


#   def get_discrete_state(self, state):
#     rounded = np.round(state * 20) / 20
#     return "_".join(map(str, rounded))


#   def simulate_episode(self):
#     is_finished = False
#     is_truncated = False

#     state, _ = self.env.reset()
#     state = self.get_discrete_state(state)
#     action = self.choose_action(state)
#     continuous_action = self.get_continuous_action(action)

#     while (not is_finished and not is_truncated):
#       new_state, reward, is_finished, is_truncated, _ = self.env.step([continuous_action])
#       new_state = self.get_discrete_state(new_state)
#       new_action = self.choose_action(new_state)

#       self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * (reward + self.gamma * self.q_table[new_state][new_action])

#       state = new_state
#       action = new_action
#       continuous_action = self.get_continuous_action(action)
  

#   def predict(self, state):
#     state = self.get_discrete_state(state)
#     action = self.choose_action(state, epsilon_override=0)
#     return [self.get_continuous_action(action)]
  

#   def save(self, path):
#     print(f"Saving model to {path}...")
#     with open(path, "wb") as file:
#       pickle.dump(self.q_table, file)
#     print("Done!")


#   def load(self, path):
#     if os.path.exists(path):
#       print(f"Loading model from {path}...")
#       with open(path, "rb") as file:
#         self.q_table = pickle.load(file)
#       print("Done!")



class NStepSarsaAgent(Agent):

  def __init__(self, env: gym.Env, alpha=0.1, gamma=0.99, epsilon=0.1, lamb=0.9, n_actions=20, state_quantise=0.05):
    super(NStepSarsaAgent, self).__init__(env)

    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.lamb = lamb
    self.n_actions = n_actions
    self.state_quantise = state_quantise
    self.q_table = {}
    self.trace_table = {}
    self.action_shape = env.action_space._shape[0]
    self.action_max = env.action_space.high[0]
    self.actions = [self.get_action_representation(list(p)) for p in product(range(self.n_actions), repeat=self.action_shape)]


  def get_continuous_action(self, representation):
    action = self.get_action_from_representation(representation)
    return (2 * self.action_max * (action + 0.5) / self.n_actions) - self.action_max
  

  def get_action_representation(self, action):
    return "_".join(map(lambda x: str(int(x)), action))
  

  def get_action_from_representation(self, representation):
    return np.array(representation.split("_"), dtype=np.float64)


  def choose_action(self, state, epsilon_override=None):
    epsilon = self.epsilon if epsilon_override is None else epsilon_override

    if state not in self.q_table:
        self.q_table[state] = {a : np.random.rand() for a in self.actions}
    
    if state not in self.trace_table:
        self.trace_table[state] = {a : 0 for a in self.actions}

    return random.choice(self.actions) if np.random.rand() < epsilon else max(self.q_table[state], key=self.q_table[state].get)
  

  def get_discrete_state(self, state):
    rounded = np.round(state / self.state_quantise) * self.state_quantise
    return "_".join(map(str, rounded))


  def simulate_episode(self):
    is_finished = False
    is_truncated = False
    reward_total = 0
    timestep = 0

    for s, actions in self.q_table.items():
        for a, _ in actions.items():
            self.trace_table[s] = {a : 0 for a in actions}

    state, _ = self.env.reset()
    state = self.get_discrete_state(state)
    action = self.choose_action(state)

    continuous_action = self.get_continuous_action(action)

    while (not is_finished and not is_truncated):
      new_state, reward, is_finished, is_truncated, _ = self.env.step(continuous_action)
      new_state = self.get_discrete_state(new_state)
      new_action = self.choose_action(new_state)

      timestep += 1
      reward_total += reward

      self.trace_table[state][action] += 1

      delta = reward + self.gamma * self.q_table[new_state][new_action] - self.q_table[state][action]

      for s, actions in self.q_table.items():
          for a, state_value in actions.items():
              trace_value = self.trace_table[s][a]
              if trace_value > 0:
                  self.q_table[s][a] = state_value + self.alpha * delta * trace_value
                  self.trace_table[s][a] = trace_value * self.gamma * self.lamb

      state = new_state
      action = new_action
      continuous_action = self.get_continuous_action(action)
    
    return timestep, reward_total
  

  def predict(self, state):
    state = self.get_discrete_state(state)
    action = self.choose_action(state, epsilon_override=0)
    return self.get_continuous_action(action)

  
  def save(self, path):
    print(f"Saving model to {path}...")
    with open(path, "wb") as file:
      pickle.dump(self.q_table, file)
    print("Done!")


  def load(self, path):
    if os.path.exists(path):
      print(f"Loading model from {path}...")
      with open(path, "rb") as file:
        self.q_table = pickle.load(file)
      print("Done!")



env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
SAVE_PATH = "sarsa_pendulum.data"

agent = NStepSarsaAgent(env)
agent.load(SAVE_PATH)
agent.train(num_timesteps=100000)
agent.save(SAVE_PATH)
agent.render()

env.close()