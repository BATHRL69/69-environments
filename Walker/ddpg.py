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

    def loss(self):
        pass

    def train(self, num_episodes=1000):
        # lines 4 - 8 and 9
        for episode in range(num_episodes):

            pass

class ActorNetwork(nn.Module):
    pass


class CriticNetwork(nn.Module):
    pass

if __name__ == "__main__":
    pass
    # set up mujoco
    # pass in the environment