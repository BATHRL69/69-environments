import torch
import torch.nn as nn

#
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/core.py
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/ddpg.py

from agent import Agent

class DDPGAgent(Agent):

    def __init__(self):
        # set hyperparams
        # ...
        # give it the open ai environment, observation
        pass

    def critic_loss(self, data):

        # extract observations from data


        #  for observations in data:

            # current_state, action, reward, next_state, d = observation

            # calculate pred = critic(current_state, action)

            # calculate loss += (pred - (r + gamma*(1 - d)*target_critic(next_state, target_actor(next_state))))

        # loss = loss / number of observations

        # return loss

        pass

    def actor_loss(self):

        
        pass

    def train(self, num_episodes=1000):
        # lines 4 - 8 and 9
        for episode in range(num_episodes):

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
    

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    pass
    # set up mujoco
    # pass in the environment