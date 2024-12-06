# Modified by ChatGPT to work with minibatches.

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
        hidden_layers=[32, 32],
        activation="Tanh",
    ):
        # TODO need to work out what network is going to work best here
        super(PPOPolicyNetwork, self).__init__()

        layers = []
        input_size = observation_space

        for layer_dimension in hidden_layers:
            layers.append(nn.Linear(input_size, layer_dimension))
            if activation == "ReLU":
                layers.append(nn.ReLU())

            elif activation == "Tanh":
                layers.append(nn.Tanh())
            input_size = layer_dimension

        layers.append(nn.Linear(hidden_layers[-1], action_space))
        self.network = nn.Sequential(*layers)
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
        """Takes in the state and action, and returns the log probability of that action, given the state. So P(A|S), assuming a normal distribution

        Args:
            state (torch.Tensor): The current state
            action (torch.Tensor): The action to be taken in this state

        Returns:
            torch.Tensor: the log-probability of action given state
        """
        action_values = self.network(state)
        distribution = Normal(action_values, self.std)
        log_probs = distribution.log_prob(action)
        return log_probs.sum(
            dim=1
        )  # Sum over action dimensions if batch processing

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
        activation="ReLU",
        hidden_layers=[32, 32],
        batch_size=64,  # New parameter for minibatch size
    ):
        super(PPOAgent, self).__init__(env)

        # Line 1 of pseudocode
        self.policy_network = PPOPolicyNetwork(
            observation_space, action_space, std, hidden_layers, activation
        )
        self.old_policy_network = PPOPolicyNetwork(
            observation_space, action_space, std, hidden_layers, activation
        )
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
        self.batch_size = batch_size  # Minibatch size

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
        state_value_estimates = self.value_network(states).squeeze()  # V(S)
        state_action_value_estimates = self.state_action_values_mc(rewards)  # Q(S,A)
        return state_action_value_estimates - state_value_estimates  # Q(S,A) - V(S)

    def advantage_estimates_gae(self, states, rewards):
        """
        Use advantage estimation on value network, using GAE
        """
        # TODO on ant V4, this returns a tensor of size length_of_trajectory - 1, which causes dimensions mismatches. I think we need to explicitly handle the terminal state, but not sure on this
        values = self.value_network(states[:-1]).squeeze()

        next_values = self.value_network(states[1:]).squeeze()

        rewards = rewards[:-1]

        deltas = rewards + self.gamma * next_values - values
        advantages = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(deltas))):
            gae = (gae * self.gamma * self.lambda_gae) + deltas[t]
            advantages[t] = gae

        # Account for the terminal state by appending a 0.
        advantages = torch.cat((advantages, torch.tensor([0.0])), dim=0)

        return advantages

    def collect_trajectories(self, num_trajectories):
        """Collect multiple trajectories for minibatch training."""
        trajectories = []
        total_rewards = []
        total_timesteps = 0

        for _ in range(num_trajectories):
            is_finished = False
            is_truncated = False
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            trajectory = []
            timesteps_in_trajectory = 0
            episode_rewards = 0

            while (
                not is_finished
                and not is_truncated
                and timesteps_in_trajectory < self.max_trajectory_timesteps
            ):
                action, _ = self.policy_network.get_action(state)
                action_np = action.detach().numpy()[0]
                new_state, reward, is_finished, is_truncated, _ = self.env.step(
                    action_np
                )
                new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
                trajectory.append(
                    (
                        state.squeeze(0),
                        action.squeeze(0).detach(),
                        reward,
                        new_state.squeeze(0),
                        is_finished,
                        is_truncated,
                    )
                )
                state = new_state
                timesteps_in_trajectory += 1
                episode_rewards += reward

            total_timesteps += timesteps_in_trajectory
            total_rewards.append(episode_rewards)
            trajectories.extend(trajectory)

        return trajectories, total_timesteps, total_rewards

    def simulate_episode(self, num_trajectories=10):
        """Simulate multiple episodes and collect trajectories."""
        self.transfer_policy_net_to_old()  # Transfer the current probability network to OLD, so that we are freezing it, before we start making updates

        # Collect multiple trajectories
        trajectories, timesteps_in_trajectory, total_rewards = self.collect_trajectories(
            num_trajectories
        )

        # Prepare tensors
        states = torch.stack([t[0] for t in trajectories])
        actions = torch.stack([t[1] for t in trajectories])
        rewards = torch.tensor([t[2] for t in trajectories], dtype=torch.float32)

        # Compute rewards to go
        rewards_to_go = self.state_action_values_mc(rewards)
        rewards_to_go = rewards_to_go.unsqueeze(1)  # For value network training

        # Compute advantage estimates
        advantage_estimates = self.advantage_estimates_mc(states, rewards)
        advantage_estimates = advantage_estimates.detach()

        # Normalize advantages
        advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / (
            advantage_estimates.std() + 1e-8
        )

        # Line 6 in Pseudocode
        num_samples = states.size(0)
        indices = np.arange(num_samples)

        # Shuffle indices for minibatch
        np.random.shuffle(indices)

        # Minibatch training
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantage_estimates[batch_indices]
            batch_rewards_to_go = rewards_to_go[batch_indices]

            network_probability_ratio = self.probability_ratios(
                batch_states, batch_actions
            )
            clipped_g = torch.clamp(
                network_probability_ratio, 1 - self.epsilon, 1 + self.epsilon
            )  # g(\epsilon, advantage estimates)
            ratio_with_advantage = batch_advantages * network_probability_ratio
            clipped_with_advantage = batch_advantages * clipped_g
            ppo_clipped = torch.min(ratio_with_advantage, clipped_with_advantage)
            policy_loss = -torch.mean(ppo_clipped)

            self.policy_optimiser.zero_grad()
            policy_loss.backward()
            self.policy_optimiser.step()

            # Update value network
            value_estimates = self.value_network(batch_states)
            value_loss = torch.mean((value_estimates - batch_rewards_to_go) ** 2)
            self.value_optimiser.zero_grad()
            value_loss.backward()
            self.value_optimiser.step()

        return timesteps_in_trajectory, sum(total_rewards)

    def train(self, num_iterations=100_000, log_iterations=100, num_trajectories=20):
        total_timesteps = 0
        total_reward = []
        average_rewards = []
        last_log = 0
        while total_timesteps < num_iterations:

            timesteps, reward = self.simulate_episode(
                num_trajectories
            )  # Collect and train on multiple trajectories
            average_episode_reward = reward / num_trajectories
            total_reward.append(average_episode_reward)
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


    def efficient_train(self, num_iterations=1000, num_trajectories=10):
        """Train our model for n iterations, without logging or plotting

        Args:
            num_iterations (int, optional): number of timesteps to run until. Defaults to 1000.
            num_trajectories (int, optional): Number of trajectories per update. Defaults to 10.

        Returns:
            array: Total rewards, for each training iteration
        """
        total_timesteps = 0
        total_rewards = []

        while total_timesteps < num_iterations:
            timesteps, reward = self.simulate_episode(
                num_trajectories
            )  # Simulate episodes and collect rewards
            total_rewards.append(reward)
            total_timesteps += timesteps

        return total_rewards

    def plot(self, average_rewards):
        """Plot the average rewards over time

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
    model.train(num_iterations=100_000, log_iterations=1000, num_trajectories=10)
    print("\n Training finished")
    print("Rendering...")
    model.render(num_timesteps=100_000)


environments = [
    {"name": "InvertedPendulum-v4", "observation_space": 4, "action_space": 1},
    {"name": "Ant-v4", "observation_space": 27, "action_space": 8},
    {"name": "Ant-v5", "observation_space": 105, "action_space": 8},
]
verbose_train(environments[0])


##################
# Testing the models
def test_models(configurations, num_iterations=50000, num_runs=3):
    results = []
    for config in configurations:
        # Create the environment
        env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")

        # Initialize the model with the given configuration
        model = PPOAgent(
            env,
            observation_space=config["observation_space"],
            action_space=config["action_space"],
            std=config["std"],
            hidden_layers=config["hidden_layers"],
            activation=config["activation"],
        )

        run_rewards = []
        print(
            f"Training with configuration: obs space={config['observation_space']}, action space={config['action_space']}, std={config['std']}, layers={config['hidden_layers']}, activation={config['activation']}"
        )

        # Train the model multiple times
        for _ in range(num_runs):
            total_rewards = model.efficient_train(num_iterations)
            run_rewards.append(
                sum(total_rewards) / len(total_rewards) if total_rewards else 0
            )

        # Calculate statistics across all runs
        average_reward = sum(run_rewards) / num_runs
        max_reward = max(run_rewards)
        min_reward = min(run_rewards)

        # Collect results
        results.append(
            {
                "configuration": config,
                "average_reward": average_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
            }
        )

        # Clean up the environment
        env.close()

    # Sort results by average reward
    sorted_results = sorted(results, key=lambda x: x["average_reward"], reverse=True)
    return sorted_results


def print_results(results):
    print("\nModel Testing Results:\n")
    for result in results:
        config = result["configuration"]
        print(
            f"Configuration - Observation Space: {config['observation_space']}, Action Space: {config['action_space']}, "
            f"Std Dev: {config['std']}, Hidden Layers: {config['hidden_layers']}, Activation: {config['activation']}"
        )
        print(
            f"Average Reward: {result['average_reward']:.2f}, Max Reward: {result['max_reward']:.2f}, Min Reward: {result['min_reward']:.2f}\n"
        )


configurations = [
    {
        "observation_space": 4,
        "action_space": 1,
        "std": 0.1,
        "hidden_layers": [64, 64],
        "activation": "ReLU",
    },
    {
        "observation_space": 4,
        "action_space": 1,
        "std": 0.2,
        "hidden_layers": [64, 64],
        "activation": "ReLU",
    },
    {
        "observation_space": 4,
        "action_space": 1,
        "std": 0.1,
        "hidden_layers": [64, 64],
        "activation": "Tanh",
    },
]

# print_results(test_models(configurations, num_iterations=10000, num_runs=3))
