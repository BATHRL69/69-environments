import gymnasium as gym
import numpy as np
import random

class SarsaAgent:

  def __init__(self, env: gym.Env, n_actions = 20, alpha=0.5, gamma=0.99, epsilon=0.1):
    self.env = env
    self.n_actions = n_actions
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.q_table = {}
  

  def get_actions(self):
    return [i for i in range(self.n_actions)]


  def get_continuous_action(self, action):
    # hard coding for now using (-3, 3) bounds on pendulum environment
    return (6 * (action + 0.5) / self.n_actions) - 3


  def choose_action(self, state, epsilon_override=None):
    epsilon = self.epsilon if epsilon_override is None else epsilon_override
    actions = self.get_actions()

    if state not in self.q_table:
        self.q_table[state] = {a : np.random.rand() for a in actions}

    return random.choice(actions) if np.random.rand() < epsilon else max(self.q_table[state], key=self.q_table[state].get)


  def get_discrete_state(self, state):
    rounded = np.round(state * 20) / 20
    return "_".join(map(str, rounded))


  def simulate_episode(self):
    is_finished = False
    is_truncated = False

    state, _ = self.env.reset()
    state = self.get_discrete_state(state)
    action = self.choose_action(state)
    continuous_action = self.get_continuous_action(action)

    while (not is_finished and not is_truncated):
      new_state, reward, is_finished, is_truncated, _ = self.env.step([continuous_action])
      new_state = self.get_discrete_state(new_state)
      new_action = self.choose_action(new_state)

      self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * (reward + self.gamma * self.q_table[new_state][new_action])

      state = new_state
      action = new_action
      continuous_action = self.get_continuous_action(action)


  def train(self, num_episodes=1000):
    ten_percent = int(num_episodes / 10)

    for i in range(num_episodes):
      self.simulate_episode()

      if i % ten_percent == 0:
        print(f"Training {10 * i / ten_percent}% complete...")
  

  def predict(self, state):
    state = self.get_discrete_state(state)
    action = self.choose_action(state, epsilon_override=0)
    return [self.get_continuous_action(action)]



class NStepSarsaAgent:

  def __init__(self, env: gym.Env, n_actions=20, alpha=0.5, gamma=0.99, epsilon=0.1, lamb=0.9):
    self.env = env
    self.n_actions = n_actions
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.lamb = lamb
    self.q_table = {}
    self.trace_table = {}


  def get_actions(self):
    return [i for i in range(self.n_actions)]


  def get_continuous_action(self, action):
    # hard coding for now using (-3, 3) bounds on pendulum environment
    return (6 * (action + 0.5) / self.n_actions) - 3


  def choose_action(self, state, epsilon_override=None):
    epsilon = self.epsilon if epsilon_override is None else epsilon_override
    actions = self.get_actions()

    if state not in self.q_table:
        self.q_table[state] = {a : np.random.rand() for a in actions}
    
    if state not in self.trace_table:
        self.trace_table[state] = {a : 0 for a in actions}

    return random.choice(actions) if np.random.rand() < epsilon else max(self.q_table[state], key=self.q_table[state].get)
  

  def get_discrete_state(self, state):
    rounded = np.round(state * 20) / 20
    return "_".join(map(str, rounded))


  def simulate_episode(self):
    is_finished = False
    is_truncated = False

    for s, actions in self.q_table.items():
        for a, _ in actions.items():
            self.trace_table[s][a] = 0

    state, _ = self.env.reset()
    state = self.get_discrete_state(state)
    action = self.choose_action(state)
    continuous_action = self.get_continuous_action(action)

    while (not is_finished and not is_truncated):
      new_state, reward, is_finished, is_truncated, _ = self.env.step([continuous_action])
      new_state = self.get_discrete_state(new_state)
      new_action = self.choose_action(new_state)

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


  def train(self, num_episodes=1000):
    ten_percent = int(num_episodes / 10)

    for i in range(num_episodes):
      self.simulate_episode()

      if i % ten_percent == 0:
        print(f"Training {10 * i / ten_percent}% complete...")
  

  def predict(self, state):
    state = self.get_discrete_state(state)
    action = self.choose_action(state, epsilon_override=0)
    return [self.get_continuous_action(action)]