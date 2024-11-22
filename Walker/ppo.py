## TODO
# Learn STD, instead of a set value.
# Update loop so that we calculate a batch of trajectories, and learn from them, instead of just one.
# Check that the old model is definetly frozen.
# Check if there is anything we need to detach in main loop that we're not detaching
# Implement rendering
# Hyper param tuning
# Speed up running
# Add doc strings
# Update size of neural network to get best results
# Try different distributions other than normal | I think normal is best for our environment.
# Try on ant

import gymnasium as gym
import numpy as np
from agent import Agent
import pickle
import os
import torch
import torch.nn as nn
from torch.distributions import Uniform, Normal
from ppo_constants import *
import torch.optim as optim
import matplotlib.pyplot as plt


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
        std=0.1,  # Standard deviation for normal distribution
    ):
        # TODO need to work out what network is going to work best here
        super(PPOPolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.Tanh(),
            nn.Linear(32, action_space),
            # nn.Tanh(),  # Ant action space is -1 to 1
        )
        self.std = std

    def get_action(self, state):
        """Given a state, gets an action, sampled from normal distribution

        Args:
            state (torch.tensor): Current state (observation space)

        Returns:
            torch.tensor: action, probability
        """
        action_values = self.network(state)
        distribution = Normal(action_values, self.std)
        action = distribution.sample()
        probability = distribution.log_prob(action)
        return action, probability

    def get_probability_given_action(self, state, action):
        action_values = self.network(state)
        distribution = Normal(action_values, self.std)
        return distribution.log_prob(action)
        # TODO will this action be in the right format for this to work?
        # We are doing exp to turn our log_prob into probability 0-1. Will do torch.exp in probability ratio method to return this to between 0 and 1

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
        super(PPOValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        return self.network(state)


class PPOAgent(Agent):
    def __init__(
        self,
        env,
        epsilon=0.2,
        gamma=0.99,
        observation_space=105,
        action_space=8,
        std=0.1,
        learning_rate=3e-4,
        weight_decay=0,
        lambda_gae=0.95,
    ):
        # Line 1 of pseudocode
        self.policy_network = PPOPolicyNetwork(observation_space, action_space, std)
        self.old_policy_network = PPOPolicyNetwork(observation_space, action_space, std)
        self.transfer_policy_net_to_old()
        self.policy_optimiser = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.value_network = PPOValueNetwork(observation_space)
        self.value_optimiser = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.env = env
        self.epsilon = (
            epsilon  # How large of a step we will take with updates used in PPO-Clip
        )
        self.gamma = gamma  # Discount factor, used in advantage estimation.
        self.max_trajectory_timesteps = (
            100000  # Maximum number of timesteps in a trajectory
        )
        self.lambda_gae = lambda_gae

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
        current_log_prob = self.policy_network.get_probability_given_action(
            state, action
        )
        old_log_prob = self.old_policy_network.get_probability_given_action(
            state, action
        )
        return torch.exp(
            current_log_prob - old_log_prob
        )  # This formula is P(A|S) / P_old(A|S) but we can do - since they are log probability

    def entropy_bonus(self):
        """
        Can add an entropy bonus to the above to encourage exploration, since this is an on-policy method
        Args:

        Returns:
        """

    def state_action_values_mc(self, rewards):
        """Compute Q value, this is the value of taking a state, and action, and then following the policy thereafter"""
        # TODO At the moment we are using monte carlo, but this could be improved with GAE
        total_reward = 0
        q_values = []
        reversed_rewards = reversed(rewards)
        for this_reward in reversed_rewards:
            total_reward = self.gamma * total_reward + this_reward
            q_values.append(total_reward)

        q_values.reverse()

        return torch.tensor(
            q_values
        )  # We have already reversed, so need to get our q values back in the order they were given

    def advantage_estimates(self, states, rewards):
        """
        Use advantage estimation on value network,
        Args:
        Returns:
        """
        state_value_estimates = torch.tensor(
            [self.value_network.forward(state) for state in states]
        )  # V(S)
        state_action_value_estimates = self.state_action_values_mc(rewards)  # Q(S,A)
        return state_action_value_estimates - state_value_estimates  # Q(S,A) - V(S)

    def advantage_estimates_gae(self, states, rewards):
        """
        Use advantage estimation on value network, using GAE
        """

        values = torch.tensor(
            [self.value_network.forward(state) for state in states[:-1]]
        )

        next_values = torch.tensor(
            [self.value_network.forward(state) for state in states[1:]]
        )

        rewards = rewards[:-1]

        deltas = rewards + self.gamma * next_values - values
        advantages = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(deltas))):
            gae = (gae * self.gamma * self.lambda_gae) + deltas[t]
            advantages[t] = gae

        return advantages

    def rewards_to_go(self):
        """Calculate rewards to go"""

        pass

    def simulate_episode(self):
        """Simulate a single episode, called by train method on parent class"""
        self.transfer_policy_net_to_old()  # Transfer the current probability network to OLD, so that we are feezing it, before we start making updates
        all_rewards = []
        is_finished = False
        is_truncated = False
        state, _ = self.env.reset()
        action, _ = self.policy_network.get_action(
            torch.tensor(state, dtype=torch.float32)
        )
        trajectories = []
        timesteps_in_trajectory = 0
        # Line 3 of pseudocode, here we are collecting a trajectory
        while (
            not is_finished
            and not is_truncated
            and timesteps_in_trajectory < self.max_trajectory_timesteps
        ):
            value = self.value_network(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            )
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            next_value = self.value_network(
                torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
            )
            trajectories.append(
                (state, action.detach(), reward, new_state, is_finished, is_truncated)
            )
            state = torch.tensor(new_state, dtype=torch.float32)
            action, _ = self.policy_network.get_action(state)
            timesteps_in_trajectory += 1

        trajectories.append(
            (state, action, reward, state, is_finished, is_truncated)
        )  # Append the final trajectory
        ## Get lists of values from our trajectories,
        states = torch.tensor(
            np.array([this_timestep[0] for this_timestep in trajectories]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            np.array([this_timestep[1] for this_timestep in trajectories]),
            dtype=torch.float32,
        )
        all_rewards.extend([this_timestep[2] for this_timestep in trajectories])
        rewards = torch.tensor(
            np.array([this_timestep[2] for this_timestep in trajectories]),
            dtype=torch.float32,
        )

        # Line 4 in pseudocode
        # Compute rewards to go TODO is this the right way to work these out using MC?
        rewards_to_go = self.state_action_values_mc(rewards)
        # Line 5 in Pseudocode
        # Compute advantage estimates
        advantage_estimates = self.advantage_estimates_gae(states, rewards)
        # Line 6 in Pseudocode
        normalisation_factor = 1 / (
            timesteps_in_trajectory
        )  # 1 / D_k T, which is just timesteps in trajectory for us because we have 1 trajectory
        network_probability_ratio = torch.stack(
            [
                self.probability_ratios(this_state, this_action)
                for this_state, this_action in zip(states, actions)
            ]
        )  # pi (a_t| s_t) / pi (a_t, s_t) #TODO this isn't working
        clipped_g = torch.clamp(
            network_probability_ratio, 1 - self.epsilon, 1 + self.epsilon
        )  # g(\epsilon, advantage estimates)
        ratio_with_advantage = advantage_estimates * network_probability_ratio
        clipped_with_advantage = advantage_estimates * clipped_g
        ppo_clipped = torch.min(ratio_with_advantage, clipped_with_advantage)
        policy_loss = -(
            normalisation_factor
            * torch.sum(
                ppo_clipped
            )  # Take the sum of ppo clipped from pseudocode, loops through every timestep/trajectory
        )  # We are minusing here because we are trying to find the arg max, so the LOSS needs to be negative. (since we are trying to minimise the loss)
        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()

        # Line 7 in pseudocode
        value_estimates = torch.tensor(
            [self.value_network.forward(this_state) for this_state in states],
            requires_grad=True,
        )
        value_loss = (normalisation_factor) * torch.mean(
            torch.square(value_estimates - rewards_to_go)
        )  # TODO pseudocode has this as SUM, but it's mean squared error. Need to workout which one works better
        value_loss.backward()
        self.value_optimiser.step()
        return timesteps_in_trajectory, sum(all_rewards)

    def train(self, num_iterations=1000, log_iterations=100):
        total_timesteps = 0
        total_reward = []
        average_rewards = []
        while total_timesteps < num_iterations:

            timesteps, reward = (
                self.simulate_episode()
            )  # Taking total reward here because we want to maximise total reward, which is keeping pendulum up
            total_reward.append(reward)
            if (
                total_timesteps % log_iterations == 0
            ):  # TODO we don't have the same number of timesteps at each step so this isn't logging every time
                average_reward = sum(total_reward) / len(total_reward)
                print(
                    f"\r Processing Progress: {(total_timesteps/ num_iterations * 100):.2f}% Average reward:{average_reward:.2f} ",
                    end="",
                    flush=True,
                )
                average_rewards.append(average_reward)
                total_reward = []
            total_timesteps += timesteps
        self.plot(average_rewards)

    def plot(self, average_rewards):
        plt.plot(average_rewards)
        plt.show()

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
model = PPOAgent(env, observation_space=4, action_space=1, std=0.2)
model.train(num_iterations=100_000)
