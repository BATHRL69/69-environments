import gymnasium as gym
import torch
import torch.nn as nn

from agent import Agent
from typing import NamedTuple, Any


class Experience(NamedTuple):
    old_state: Any
    new_state: Any
    action: Any
    reward: float
    is_terminal: bool


class SACPolicyNetwork(nn.Module):
    def __init__(self):
        super(SACPolicyNetwork,self).__init__()
        self._input_dim = 100 #change these values later
        self.ann = nn.Sequential(
            nn.Linear(self._embeddings_dim, self._mlp_input_dim),
            nn.ReLU(),
            nn.Linear(self._mlp_input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self,input:torch.Tensor)->torch.Tensor:
        x = self.ann(input)
        return x


class SACValueNetwork(nn.module):
    def __init__(self):
        pass


class SACAgent(Agent):
    def __init__(self, env: gym.Env, update_threshold: int):
        self.env = env
        self.update_threshold = update_threshold
        self.replay_buffer = []

        self.actor = SACPolicyNetwork()
        self.critic_1 = SACValueNetwork()
        self.critic_2 = SACValueNetwork()
        self.critic_target_1 = SACValueNetwork()
        self.critic_target_2 = SACValueNetwork()

    
    def simulate_episode(self):
        is_finished = False
        is_truncated = False

        state, _ = self.env.reset()
        timestep = 0

        while True:
            timestep += 1

            action = self.actor.forward(state, train=False)
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            self.replay_buffer.append(Experience(state, new_state, action, reward, is_finished))

            if is_finished or is_truncated:
                break

            if timestep % self.update_threshold != 0:
                continue

            # update



    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError