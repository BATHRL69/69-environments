import torch
import torch.nn as nn
import gymnasium as gym

#
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/core.py
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/ddpg.py

from agent import Agent

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

    def predict(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()

    def loss(self):
        pass

    def train(self, num_episodes=1000):
        # lines 4 - 9 in https://spinningup.openai.com/en/latest/algorithms/ddpg.html#documentation-pytorch-version
        for episode in range(num_episodes):

            pass

class ActorNetwork(nn.Module):

    def __init__(self,  act_dim, obs_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class CriticNetwork(nn.Module):
    pass

if __name__ == "__main__":

    env = gym.make("Humanoid-v4", render_mode=None)
    agent = DDPGAgent(env.action_space, env.observation_space)