import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from typing import NamedTuple, Any


class Experience(NamedTuple):
    old_state: Any
    new_state: Any
    action: Any
    reward: float
    is_terminal: bool


class ReplayBuffer():
    def __init__(self, max_capacity: int):
        self.max_capacity = max_capacity
        self.buffer = np.array([Experience(None, None, None, 0, False) for _ in range(max_capacity)])
        self.counter = 0
        
    
    def add(self, experience: Experience):
        idx = self.counter % self.max_capacity
        self.buffer[idx] = experience
        self.counter += 1
            
    
    def get(self, batch_size: int) -> list[Experience]:
        valid_entries = min(self.counter, self.max_capacity)
        return np.random.sample(self.buffer[:valid_entries], batch_size)


class SACPolicyLoss(nn.Module):
    def __init__(self):
        super(SACPolicyLoss, self).__init__()
    
    def forward(self, min_critic, entropy, alpha):
        return min_critic - alpha * entropy


class SACPolicyNetwork(nn.Module):
    def __init__(self):
        super(SACPolicyNetwork,self).__init__()
        self._input_dim = 256
        self.ann = nn.Sequential(
            nn.Linear(self._input_dim, self._input_dim),
            nn.ReLU(),
            nn.Linear(self._input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self,input:torch.Tensor)->torch.Tensor:
        action = self.ann(input)#samples a prob dis
        return action
    
    def get_probability(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        probability = self.ann(state, action) # get probability of picking action from state
        return probability


class SACValueNetwork(nn.Module):
    def __init__(self):
        super(SACPolicyNetwork,self).__init__()
        self._input_dim = 256
        self.ann = nn.Sequential(
            nn.Linear(self._input_dim, self._input_dim),
            nn.ReLU(),
            nn.Linear(self._input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        value_estimate = self.ann(state, action) # estimate the value of an action in a state
        return value_estimate


class SACAgent(Agent):
    # TODO: count timesteps across episodes, not within - agent still needs to learn on short episodes

    def __init__(self, env: gym.Env, update_threshold: int = 100, batch_size: int = 256, alpha: float = 1e-4, gamma: float = 0.99, polyak = 0.99):
        self.env = env
        self.update_threshold = update_threshold
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak
        self.replay_buffer = ReplayBuffer(1e6)

        self.actor = SACPolicyNetwork()
        self.critic_1 = SACValueNetwork()
        self.critic_2 = SACValueNetwork()
        self.critic_target_1 = SACValueNetwork()
        self.critic_target_2 = SACValueNetwork()

        self.actor_loss = SACPolicyLoss()
        self.critic_1_loss = nn.MSELoss()
        self.critic_2_loss = nn.MSELoss()

        self.actor_optimiser = optim.Adam(self.actor.parameters())
        self.critic_1_optimiser = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimiser = optim.Adam(self.critic_2.parameters())

        self.polyak_update(0)


    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)
        
        for (parameter, target_parameter) in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)

    
    def simulate_episode(self):
        is_finished = False
        is_truncated = False

        state, _ = self.env.reset()
        timestep = 0

        while True:
            timestep += 1

            action = self.actor.forward(state, train=False)
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            self.replay_buffer.add(Experience(state, new_state, action, reward, is_finished))

            if is_finished or is_truncated:
                break

            if timestep % self.update_threshold != 0:
                continue

            for experience in self.replay_buffer.get(self.batch_size):
                # line 12 of pseudocode
                actor_prediction_new = self.actor.forward(experience.new_state, train=False)
                critic_target_1_prediction = self.critic_target_1.forward(experience.new_state, actor_prediction_new, train=False)
                critic_target_2_prediction = self.critic_target_2.forward(experience.new_state, actor_prediction_new, train=False)
                critic_target_clipped = min(critic_target_1_prediction, critic_target_2_prediction)
                predicted_target_reward = critic_target_clipped - self.alpha * np.log(self.actor.get_probability(experience.new_state, actor_prediction_new))
                target = reward + self.gamma * (1 - experience.is_terminal) * predicted_target_reward

                # line 13 of pseudocode
                # TODO: batch norm
                critic_1_evaluation = self.critic_1.forward(experience.old_state, experience.action, train=False)
                critic_2_evaluation = self.critic_2.forward(experience.old_state, experience.action, train=False)

                critic_1_loss = self.critic_1_loss(target, critic_1_evaluation)
                critic_2_loss = self.critic_2_loss(target, critic_2_evaluation)

                self.critic_1_optimiser.zero_grad()
                self.critic_2_optimiser.zero_grad()

                critic_1_loss.backward()
                critic_2_loss.backward()

                self.critic_1_optimiser.step()
                self.critic_2_optimiser.step()

                # line 14 of pseudocode
                # TODO: batch norm
                actor_prediction_old = self.actor.forward(experience.old_state, train=False)
                critic_1_prediction = self.critic_1.forward(experience.old_state, actor_prediction_old, train=False)
                critic_2_prediction = self.critic_2.forward(experience.old_state, actor_prediction_old, train=False)
                critic_clipped = min(critic_1_prediction, critic_2_prediction)
                entropy = np.log(self.actor.get_probability(experience.old_state, actor_prediction_old))

                actor_loss = self.actor_loss(critic_clipped, entropy, self.alpha)

                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                self.actor_optimiser.step()

                # line 15 of pseudocode
                self.polyak_update(self.polyak)


    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError