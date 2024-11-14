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

from enum import Enum
from ppo_constants import *


class PPOPolicyNetwork(nn.Module):
    """This is sometimes also called the ACTOR network

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        pass


class PPOValueNetwork(nn.Module):
    """This is sometimes also called the CRITIC network

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        pass


class PPOAgent(Agent):
    def __init__(self, env, epsilon=0.001, gamma=0.99):
        self.policy_network = None
        self.value_network = None
        self.env = env
        self.epsilon = epsilon  # How large of a step we will take with updates
        self.gamma = gamma  # Discount factor

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

    def train(self):
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
