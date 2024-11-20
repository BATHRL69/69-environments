import torch
import torch.nn as nn
import gymnasium as gym
import copy
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
        self.buffer = []

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

    def __init__(self, action_space, observation_space):
        # set hyperparams
        # polyak, learning rate, num episodes etc

        self.action_space = action_space
        self.observation_space = observation_space

        action_dim: int = action_space.shape[0]
        act_limit: int = action_space.high[0]
        observation_dim: int = observation_space.shape[0]

        # policy
        self.actor = ActorNetwork(action_dim, observation_dim)

        # q-value function
        self.critic = CriticNetwork(observation_dim, action_dim)


        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

    def predict(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()

    def critic_loss(self, data):
        # extract observations from data
        i = 0
        for observation in data:
            current_state, action, reward, next_state, terminal = observation
            pred = self.critic(current_state, action)
            loss += (pred - (reward+ self.gamma*(1 - terminal)*self.target_critic(next_state, self.target_actor(next_state))))
            i += 1
            
        if (i != 0):
            loss = loss / i

        return loss

    def actor_loss(self, data):
        i = 0
        for observation in data: 
            current_state, action, reward, next_state, terminal = observation
            loss = -self.critic(current_state, self.actor(current_state))

        if (i != 0):
            loss = loss / i

        return loss 

    def train(self, num_train_episodes=1000, start_steps=10000):

        # do randomly steps

        # lines 4 - 9 in https://spinningup.openai.com/en/latest/algorithms/ddpg.html#documentation-pytorch-version
        for episode in range(num_train_episodes):

            pass

class ActorNetwork(nn.Module):

    def __init__(self, hidden_size, activation, action_dim, state_dim):
        super().__init__()
        input_size = state_dim
        output_size = action_dim
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

if __name__ == "__main__":

    env = gym.make("Humanoid-v4", render_mode=None)
    agent = DDPGAgent(env.action_space, env.observation_space)