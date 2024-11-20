import torch
import torch.nn as nn
import gymnasium as gym
import copy
from typing import Tuple, List
import numpy as np
from torch.optim import Adam

#
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/core.py
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/ddpg.py

from agent import Agent
import random
random.seed(0)

class ReplayBuffer:

    def __init__(self, max_buffer_size, sample_size):
        self.max_buffer_size = max_buffer_size
        self.sample_size = sample_size
        self.buffer: List[Tuple] = []

    def add(self, observation):
        if len(self.buffer) >= self.max_buffer_size:
            # randomly select an index
            index = random.randint(0, len(self.buffer) - 1)
            self.buffer[index] = observation
        else:
            self.buffer.append(observation)

    def sample(self):
        sample = []
        indices_selected = set()
        for i in range(self.sample_size):
            index = random.randint(0, len(self.buffer) - 1)
            while index in indices_selected:
                index = random.randint(0, len(self.buffer) - 1)
            sample.append(self.buffer[index])
        return sample

class DDPGAgent(Agent):

    def __init__(
            self,
            action_space,
            observation_space,
            max_buffer_size: int,
            replay_sample_size: int,
            actor_lr: float,
            critic_lr: float,
            polyak: float = 0.995,
            gamma: float = 0.99
    ):
        # set hyperparams
        self.max_buffer_size = max_buffer_size
        self.replay_sample_size = replay_sample_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.gamma = gamma

        self.action_space = action_space
        self.observation_space = observation_space

        action_dim: int = action_space.shape[0]
        act_limit: int = action_space.high[0]
        observation_dim: int = observation_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.max_buffer_size, self.replay_sample_size)

        # policy
        self.actor = ActorNetwork(action_dim, observation_dim)
        self.actor_optimiser = Adam(self.actor.parameters(), lr = self.actor_lr)

        # q-value function
        self.critic = CriticNetwork(observation_dim, action_dim)
        self.critic_optimiser = Adam(self.critic.parameters(), lr = self.critic_lr)

        # target
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # freeze target actor and target critic
        for parameter in self.target_actor.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

    def predict(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()

    def critic_loss(self, data):
        # extract observations from data
        i = 0
        loss = 0
        for observation in data:
            current_state, action, reward, next_state, terminal = observation
            pred = self.critic(current_state, action)
            loss += (pred - (reward+ self.gamma*(1 - terminal)*self.target_critic.get_q_value(next_state, self.target_actor.get_action(next_state))))
            i += 1
            
        if (i != 0):
            loss = loss / i

        return loss

    def actor_loss(self, data):
        i = 0
        loss = 0
        for observation in data: 
            current_state, action, reward, next_state, terminal = observation
            loss = -self.critic.get_q_value(current_state, self.actor.get_action(current_state))

        if (i != 0):
            loss = loss / i

        return loss 

    def train(self, num_train_episodes=1000, start_steps=10000):

        # do randomly steps
        for episode in range(start_steps):
            pass

        # lines 4 - 9 in https://spinningup.openai.com/en/latest/algorithms/ddpg.html#documentation-pytorch-version
        for episode in range(num_train_episodes):

            pass

    def update_weights(self):
        samples = self.replay_buffer.sample()
        self.actor_optimiser.zero_grad()
        self.critic_optimiser.zero_grad()

        # compute critic loss
        critic_loss = self.critic_loss(samples)
        critic_loss.backward()
        self.critic_optimiser.step()

        # freeze the crtic
        for parameter in self.critic.parameters():
            parameter.requires_grad = False

        # actor loss computation
        actor_loss = self.actor_loss(samples)
        actor_loss.backward()
        self.actor_optimiser.step()

        # unfreeze the critic
        for parameter in self.critic.parameters():
            parameter.requires_grad = True

        # polyak averaging -> actor params
        for actor_p, target_actor_p in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_p.data.mul_(self.polyak)
            target_actor_p.data.add_((1 - self.polyak) * actor_p.data)

        # polyak averaging -> critic params
        for critic_p, target_critic_p in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_p.data.mul_(self.polyak)
            target_critic_p.data.add_((1 - self.polyak) * critic_p.data)

class ActorNetwork(nn.Module):

    def __init__(self, hidden_size, activation, action_dim, state_dim, action_lim_high, action_lim_low):
        super().__init__()
        self.action_lim_high = action_lim_high
        self.action_lim_low = action_lim_low
        input_size = state_dim
        output_size = action_dim
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size[0])) # input layer
        layers.append(activation)

        for i in range(0, len(hidden_size) - 2):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size[-1], output_size)) # output layer
        layers.apppend(nn.t)

        self.network = nn.Sequential(*layers) # unpack layers and activation functions into sequential

    def forward(self, x):

        x = torch.tanh(self.network(x)) # we need to scale the network's output within the action range
        action_range = (self.action_max - self.action_min) / 2.0
        action_mid = (self.action_max + self.action_min) / 2.0
        scaled_output = action_mid + action_range * x
        return scaled_output
    

    def get_action(self, state):
        
        return self.forward(state)


class CriticNetwork(nn.Module):
        
    def __init__(self, hidden_size, activation, action_dim, state_dim):
        super().__init__()
        input_size = action_dim + state_dim
        output_size = 1 # critic network just outputs a value
        layers = []

        layers.append(nn.Linear(input_size, hidden_size[0])) # input layer
        layers.append(activation)

        for i in range(0, len(hidden_size) - 2):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size[-1], output_size)) # output layer

        self.network = nn.Sequential(*layers) # unpack layers and activation functions into sequential

    def forward(self, x):
        return self.network(x)
    
    def get_q_value(self, state, action):
        x = np.concat(state, action) # ?? should be numerised
        return self.forward(x)

if __name__ == "__main__":

    env = gym.make("Humanoid-v4", render_mode=None)
    agent = DDPGAgent(env.action_space, env.observation_space)