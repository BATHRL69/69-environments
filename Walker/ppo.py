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
import torch.optim as optim


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

    def get_action(self, state):
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

    def get_probability_given_action(self, state, action):
        action_values = self.network(state)
        distribution = Normal(action_values, 1.0)
        return torch.exp(
            distribution.log_prob(action)
        )  # TODO will this action be in the right format for this to work?
        # TODO We are doing exp to turn our log_prob into probability 0-1. But do we actually want log_prob?

    def evaluate(self, state, action):
        pass

    def forward(self, state):
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

    def forward(self, state):
        return self.network(state)


class PPOAgent(Agent):
    def __init__(
        self,
        env,
        epsilon=0.001,
        gamma=0.99,
    ):
        # Line 1 of pseudocode
        self.policy_network = PPOPolicyNetwork()
        self.old_policy_network = PPOPolicyNetwork()
        self.policy_optimiser = optim.Adam(self.policy_network.parameters())
        self.value_network = PPOValueNetwork()
        self.value_optimiser = optim.Adam(self.actor_network.parameters())
        self.env = env
        self.epsilon = (
            epsilon  # How large of a step we will take with updates used in PPO-Clip
        )
        self.gamma = gamma  # Discount factor
        self.max_trajectory_timesteps = (
            1000  # Maximum number of timesteps in a trajectory
        )

    def transfer_policy_net_to_old(self):
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

    def probability_ratios(self, state, action):
        """Calculate our probability ratio , which is:
        Ratio = new_policy(state,action) / old_policy(state,action)
        Args:
            state (_type_): S_t
            action (_type_): A_t

        Returns:
            _type_: probability ratio
        """

        return None

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

    def rewards_to_go(self):
        """Calculate rewards to go"""

        pass

    def simulate_episode(self):
        """Simulate a single episode, called by train method on parent class"""
        self.transfer_policy_net_to_old()  # Transfer the current probability network to OLD, so that we are feezing it, before we start making updates

        is_finished = False
        is_truncated = False
        state, _ = self.env.reset()
        action, _ = self.policy_network.get_action(state)
        trajectories = []
        timesteps_in_trajectory = 0
        # Line 3 of pseudocode, here we are collecting a trajectory
        while (
            not is_finished
            and not is_truncated
            and timesteps_in_trajectory < self.max_trajectory_timesteps
        ):
            new_state, reward, is_finished, is_truncated, _ = self.env.step([action])
            trajectories.append(
                (state, action, reward, new_state, is_finished, is_truncated)
            )
            state = new_state
            action, _ = self.policy_network.get_action(state)
            timesteps_in_trajectory += 1
        # Line 4 in pseudocode
        # Compute rewards to go
        rewards_to_go = torch.tensor()  # TODO
        # Line 5 in Pseudocode
        # Compute advantage estimates
        advantage_estimates = torch.tensor()  # TODO A^\pi _\theta_k (s_t, a_t)

        # Line 6 in Pseudocode
        normalisation_factor = 1 / (
            timesteps_in_trajectory
        )  # 1 / D_k T, which is just timesteps in trajectory for us because we have 1 trajectory
        ratio = torch.tensor()  # pi (a_t| s_t) / pi (a_t, s_t)
        clipped_g = torch.clamp(
            advantage_estimates, 1 - self.epsilon, 1 + self.epsilon
        )  # g(\epsilon, advantage estimates)
        ratio_with_advantage = advantage_estimates * ratio
        ppo_clipped = torch.min(ratio_with_advantage, clipped_g)
        loss = normalisation_factor * ppo_clipped
        self.policy_optimiser.zero_grad()

        loss.backward()

        self.policy_network.step()

    def train(self, num_iterations=10):
        for _ in range(num_iterations):
            self.simulate_episode()
            print("...")

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


env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
model = PPOAgent(env)
model.train(num_iterations=10)
