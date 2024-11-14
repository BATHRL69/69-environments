## TODO
# Skeleton code - Solly
#
#
#

import gymnasium as gym
import numpy as np
import random
from agent import Agent
import pickle
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
from enum import Enum
from ppo_constants import *


class PPOPolicyNetwork(nn.Module):
    """This is sometimes also called the ACTOR network.
    Basically, the goal of this network is to take in a state, and give us the best possible actions to take in that state.
    We will do this with a neural net that takes in the observation space, and outputs action space values.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        observation_space=105,  # Defaults set from ant walker
        action_space=8,  # Defaults set from ant walker
    ):
        # TODO need to work out what network is going to work best here
        # TODO need to workout what size of layer works best#

        self.network = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.Tanh(),
            nn.linear(32, action_space),
            nn.Tanh(),  # Ant action space is -1 to 1
        )

    def action(self, state):
        """Given a state, gets an action, sampled from normal distribution

        Args:
            state (torch.tensor): Current state (observation space)

        Returns:
            torch.tensor: action, probability
        """
        action_values = self.network(state)
        distribution = Normal(action_values, 1.0)  # Std of 1.0
        action = distribution.sample()
        probability = distribution.log_prob(action)
        return action, probability

    def evaluate(self, state, action):
        pass

    def forward(self, state):
        raise NotImplementedError()
        return self.network(state)


class PPOValueNetwork(nn.Module):
    """This is sometimes also called the CRITIC network.
    Basically, this network will give us the predicted return for a single state (observation space), given all of our observations

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        observation_space=105,  # Defaults set from ant walker
    ):
        self.network = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.linear(32, 1),
        )


class PPOAgent(Agent):
    def __init__(
        self,
        env,
        epsilon=0.001,
        gamma=0.99,
    ):
        self.policy_network = None
        self.value_network = None
        self.env = env
        self.epsilon = epsilon  # How large of a step we will take with updates
        self.gamma = gamma  # Discount factor
        self.optimizer = torch.optim

    def probability_ratios(self, state, action):
        """Calculate our probability ratio, which is:
        Ratio = new_policy(state,action) / old_policy(state,action)
        Args:
            state (_type_): S_t
            action (_type_): A_t

        Returns:
            _type_: probability ratio
        """

        return None

    def update_policy(self):
        """
        Update our policy network using SGD to minimise the value of clip function
        Args:

        Returns:


        """
        pass

    def entropy_bonus(self):
        """
        Can add an entropy bonus to the above to encourage exploration, since this is an on-policy method
        Args:

        Returns:

        """

    def advantage_estimates(self):
        """
        Use advantage estimation on value network
        Args:

        Returns:

        """

    def update_value(self):
        pass

    def create_episode(self):
        pass

    def ppo_clip(self, state, action):
        pass

    def simulate_episode(self):
        """Simulate a single episode, called by train method on parent class"""
        pass

    def predict(self):
        pass

    def save(self, path):
        """Pickle save our policy and value_networks
        Args:
            path (str): Path to save policy and value networks
        """
        print(f"Saving model to {path}...")
        with open(path + PickleLocations.POLICY_NETWORK.value, "wb") as file:
            pickle.dump(self.policy_network, file)
        with open(path + PickleLocations.VALUE_NETWORK.value, "wb") as file:
            pickle.dump(self.value_network, file)

    def load(self, path):
        """Pickle load our policy and value networks

        Args:
            path (str): Path to load policy and value networks
        """

        if os.path.exists(path):
            print(f"Loading model from {path}...")
            with open(path + PickleLocations.POLICY_NETWORK.value, "rb") as file:
                self.policy_network = pickle.load(file)
            with open(path + PickleLocations.VALUE_NETWORK.value, "rb") as file:
                self.value_network = pickle.load(file)
