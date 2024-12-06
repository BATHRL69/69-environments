## TODO
# Learn STD, instead of a set value.
# Update loop so that we calculate a batch of trajectories, and learn from them, instead of just one.
# Check that the old model is definetly frozen.
# Check if there is anything we need to detach in main loop that we're not detaching
# Hyper param tuning
# Speed up running
# Update size of neural network to get best results
# Try different distributions other than normal | I think normal is best for our environment | Stable baslines uses DiagGaussianDistribution or StateDependentNoiseDistribution, so these could be ones to try, rllib uses normal.
# Try on ant
# Add entropy bonus

import gymnasium as gym
import numpy as np
from agent import Agent
import pickle
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
from ppo_constants import *
import torch.optim as optim
import matplotlib.pyplot as plt


class PPOPolicyNetwork(nn.Module):
    """This is sometimes also called the ACTOR network.
    Basically, the goal of this network is to take in a state, and give us the best possible actions to take in that state.
    We will do this with a neural net that takes in the observation space, and outputs action space values.
    """

    def __init__(
        self,
        observation_space=27,  # Defaults set from ant walker
        action_space=8,  # Defaults set from ant walker
        std=0.1,  # Standard deviation for normal distribution
    ):
        # TODO need to work out what network is going to work best here
        super(PPOPolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_space),
        )
        self.std = std
        self.log_std = nn.Parameter(torch.full((action_space,), np.log(std)))


    def _get_distribution(self, state):
        action_values = self.network(state)
        std = torch.exp(self.log_std)
        return Normal(action_values, std)

    def get_action(self, state):
        """Given a state, gets an action, sampled from normal distribution

        Args:
            state (torch.tensor): Current state (observation space)

        Returns:
            torch.tensor: action, probability
        """
        #action_values = self.network(state)
        distribution = self._get_distribution(state)
        action = distribution.sample()
        probability = distribution.log_prob(action).sum(dim=-1)
        return action, probability

    def get_probability_given_action(self, state, action):
        """Takes in the state and action, and returns the log probability of that action, given the state. So P(A|S), assuming a normal distribution

        Args:
            state (torch.Tensor): The current state
            action (torch.Tensor): The action to be taken in this state

        Returns:
            torch.Tensor: the log-probability of action given state
        """
        #action_values = self.network(state)
        distribution = self._get_distribution(state)
        log_probs = distribution.log_prob(action).sum(dim=-1)
         
        return log_probs  # We can do sum here because they are LOG probs, usually our conditional probability would be x * y.
        # We are doing exp to turn our log_prob into probability 0-1. Will do torch.exp in probability ratio method to return this to between 0 and 1

    def evaluate(self, state, action):
        pass

    def forward(self, state):
        return self.network(state)


class PPOValueNetwork(nn.Module):
    """This is sometimes also called the CRITIC network.
    Basically, this network will give us the predicted return for a single state (observation space), given all of our observations

    """

    def __init__(
        self,
        observation_space=27,  # Defaults set from ant walker
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
        observation_space=27,  # Default from ant-v4
        action_space=8,
        std=0.1,
        learning_rate=3e-4,
        weight_decay=0,
        lambda_gae=0.95,
        minibatch_size=256,
        num_trajectories=10,
        num_epochs=3,
    ):
        super(PPOAgent, self).__init__(env)
        
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
        self.epsilon = (
            epsilon  # How large of a step we will take with updates used in PPO-Clip
        )
        self.gamma = gamma  # Discount factor, used in advantage estimation.
        self.max_trajectory_timesteps = (
            100000  # Maximum number of timesteps in a trajectory
        )
        self.lambda_gae = lambda_gae
        self.minibatch_size = minibatch_size
        self.num_trajectories = num_trajectories
        self.num_epochs = num_epochs

    def transfer_policy_net_to_old(self):
        """Copies our current policy into self.old_policy_network"""
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

    def probability_ratios(self, state, action):
        """Calculate our probability ratio , which is:
        Ratio = new_policy(state,action) / old_policy(state,action)
        Args:
            state (torch.Tensor): S_t, the state at timestep t
            action (torch.Tensor): A_t, the action to calculate probability ratio for, at time step t

        Returns:
            torch.Tensor: probability ratio of action, given state.
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
        """Compute Q value, this is the value of taking a state, and action, and then following the policy thereafter
        Args:
            rewards (torch.Tensor): a list of the rewards in our trajectory
        Returns:
            torch.Tensor: A tensor, list of state action rewards for each step in our trajectory
        """
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

    def advantage_estimates_mc(self, states, rewards):
        """
        Takes in states and corresponding rewards, and calculate the advantage estimates for each state.
        This advantage estimate tells us how much better the state action pairs in our trajectory where than current.
        This is done by subtracting the value of taking action A in state S, and following the policy thereafter, from just following the policy in state.
        Q(S,A) is calculated using MC estimation at the moment, V(S) is from our value/critic network.
        Q(S,A) - V(S)
        Args:
            states (torch.Tensor) :
            rewards (torch.Tensor) :
        Returns:
            torch.Tensor: The advantage estimate for each step in trajectory.
        """
        state_value_estimates = torch.tensor(
            [self.value_network.forward(state) for state in states]
        )  # V(S)
        state_action_value_estimates = self.state_action_values_mc(rewards)  # Q(S,A)
        return state_action_value_estimates - state_value_estimates  # Q(S,A) - V(S)

    # def advantage_estimates_gae(self, states, rewards):
    #     """
    #     Use advantage estimation on value network, using GAE
    #     """
    #     # TODO on ant V4, this returns a tensor of size length_of_trajectory - 1, which causees dimensions mismatches. I think we need to explicitly handle the terminal state, but not sure on this
    #     values = torch.tensor(
    #         [self.value_network.forward(state) for state in states[:-1]]
    #     )

    #     next_values = torch.tensor(
    #         [self.value_network.forward(state) for state in states[1:]]
    #     )

    #     rewards = rewards[:-1]

    #     deltas = rewards + self.gamma * next_values - values
    #     advantages = torch.zeros_like(rewards)

    #     gae = 0
    #     for t in reversed(range(len(deltas))):
    #         gae = (gae * self.gamma * self.lambda_gae) + deltas[t]
    #         advantages[t] = gae

    #     # Account for the terminal state by appending a 0.
    #     advantages = torch.cat((advantages, torch.tensor([0.0])), dim=0)

    #     return advantages

    def get_trajectories(self):
        trajectories = []

        timesteps_count = 0
        for _ in range(self.num_trajectories):
            is_finished = False
            is_truncated = False
            state, _ = self.env.reset()
            action, _ = self.policy_network.get_action(
                torch.tensor(state, dtype=torch.float32)
            )
            trajectory = []
            # Line 3 of pseudocode, here we are collecting a trajectory
            while not is_finished and not is_truncated:
                new_state, reward, is_finished, is_truncated, _ = self.env.step(
                    action.detach().numpy()
                )
                trajectory.append(
                    (
                        state,
                        action.detach(),
                        reward,
                        new_state,
                        is_finished,
                        is_truncated,
                    )
                )
                state = torch.as_tensor(new_state, dtype=torch.float32).flatten()
                action, _ = self.policy_network.get_action(state)
                timesteps_count += 1

            trajectory.append(
                (state, action, reward, state, is_finished, is_truncated)
            )  # Append the final trajectory
            trajectories.extend(trajectory)
        return trajectories, timesteps_count

    def update_params(
        self, rewards_to_go, advantage_estimates, states, actions, total_timesteps
    ):
        # Line 6 in Pseudocode
        normalisation_factor = 1 / (
            total_timesteps
        )  # 1 / D_k T, which is just timesteps in trajectory for us because we have 1 trajectory
        network_probability_ratio = (
            torch.stack(  # Stack all the values in our list together,into one big list
                [
                    self.probability_ratios(this_state, this_action)
                    for this_state, this_action in zip(states, actions)
                ]
            )
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
        value_loss = normalisation_factor * torch.mean(
            torch.square(value_estimates - rewards_to_go)
        )
        value_loss.backward()
        self.value_optimiser.step()

    def simulate_episode(self):
        """Simulate a single episode, called by train method on parent class"""
        self.transfer_policy_net_to_old()  # Transfer the current probability network to OLD, so that we are feezing it, before we start making updates

        trajectories, total_timesteps = self.get_trajectories()

        ## Get lists of values from our trajectories,
        states = torch.tensor(
            np.array([this_timestep[0] for this_timestep in trajectories]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            np.array([this_timestep[1] for this_timestep in trajectories]),
            dtype=torch.float32,
        )
        rewards = torch.tensor(
            np.array([this_timestep[2] for this_timestep in trajectories]),
            dtype=torch.float32,
        )

        # Line 4 in pseudocode
        # Compute rewards to go TODO is this the right way to work these out using MC?
        rewards_to_go = self.state_action_values_mc(rewards)
        # Line 5 in Pseudocode
        # Compute advantage estimates
        advantage_estimates = self.advantage_estimates_mc(states, rewards)
        for _ in range(self.num_epochs):
            for current_batch_start in range(0, total_timesteps, self.minibatch_size):
                current_batch_end = current_batch_start + self.minibatch_size
                batch_locations = np.arange(total_timesteps)
                np.random.shuffle(batch_locations)
                current_batch = batch_locations[current_batch_start:current_batch_end]
                self.update_params(
                    rewards_to_go[current_batch],
                    advantage_estimates[current_batch],
                    states[current_batch],
                    actions,
                    total_timesteps,
                )
        return total_timesteps, torch.sum(rewards)

    def train(self, num_iterations=100_000, log_iterations=100):
        """Train our agent for "n" timesteps

        Args:
            num_iterations (int, optional): The number of timesteps to run until. Defaults to 100_000.
            log_iterations (int, optional): Logs after every n timesteps. Defaults to 100.
        """
        total_timesteps = 0
        total_reward = []
        average_rewards = []
        last_log = 0
        while total_timesteps < num_iterations:

            timesteps, reward = (
                self.simulate_episode()
            )  # Taking total reward here because we want to maximise total reward, which is keeping pendulum up
            total_reward.append(reward / self.num_trajectories)

            if total_timesteps - last_log > log_iterations:
                average_reward = sum(total_reward) / len(total_reward)
                print(
                    f"\r Processing Progress: {(total_timesteps/ num_iterations * 100):.2f}% Average reward:{average_reward:.2f} ",
                    end="",
                    flush=True,
                )

                average_rewards.append(average_reward)
                total_reward = []
                last_log = total_timesteps
            total_timesteps += timesteps
        self.plot(average_rewards)

    def efficient_train(self, num_iterations=1000):
        """Train our model for n iterations, without logging or plotting

        Args:
            num_iterations (int, optional): number of timesteps to run until. Defaults to 1000.

        Returns:
            array: Total rewards, for each training iteration
        """
        total_timesteps = 0
        total_rewards = []

        while total_timesteps < num_iterations:
            timesteps, reward = (
                self.simulate_episode()
            )  # Simulate an episode and collect rewards
            total_rewards.append(reward)
            total_timesteps += timesteps

        return total_rewards

    def plot(self, average_rewards):
        """Plot the average rewards other time

        Args:
            average_rewards (array): The average reward for each logging timestep
        """
        plt.plot(average_rewards)
        plt.show()

    def predict(self, state):
        """Predict the best action

        Args:
            state (torch.Tensor): The state to predict best action for

        Returns:
            torch.Tensor: The best action to take in state S
        """
        with torch.no_grad():
            action = self.policy_network.forward(state)
        return action

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


def verbose_train(environment):
    """Train our model with progress updates and rendering

    Args:
        environment (array): The environment to train model on, should include name, observation_space, and action_space
    """
    env = gym.make(environment["name"], render_mode="rgb_array")
    model = PPOAgent(
        env,
        observation_space=environment["observation_space"],
        action_space=environment["action_space"],
        std=0.1,
    )
    model.train(num_iterations=100_000, log_iterations=1000)
    print("\n Training finished")
    print("Rendering...")
    model.render(num_timesteps=100_000)


environments = [
    {"name": "InvertedPendulum-v4", "observation_space": 4, "action_space": 1},
    {"name": "Ant-v4", "observation_space": 27, "action_space": 8},
    {"name": "Ant-v5", "observation_space": 105, "action_space": 8},
]
verbose_train(environments[1])
