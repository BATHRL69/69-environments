## TODO
# Hyper param tuning
# Speed up running
# Update size of neural network to get best results
# Try different distributions other than normal | I think normal is best for our environment | Stable baslines uses DiagGaussianDistribution or StateDependentNoiseDistribution, so these could be ones to try, rllib uses normal.

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
import random

GLOBAL_TIMESTEPS = []
GLOBAL_REWARDS = []


class PPOPolicyNetwork(nn.Module):
    """This is sometimes also called the ACTOR network.
    Basically, the goal of this network is to take in a state, and give us the best possible actions to take in that state.
    We will do this with a neural net that takes in the observation space, and outputs action space values.
    """

    def __init__(
        self,
        observation_space=27,  # Defaults set from ant walker
        action_space=8,  # Defaults set from ant walker
        std=0.4,  # Standard deviation for normal distribution
    ):
        super(PPOPolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_space)
        self.log_std = nn.Linear(256, action_space)
        self.action_max = 1

    # def update_std(self, timesteps_through, total_timesteps):
    #     """Update STD over training, to encourage exploration at the start, and then hone in on correct actions neared the end

    #     Args:
    #         timesteps_through (int): Amount of timesteps through at curent point in training
    #         total_timesteps (int): Total timesteps for training
    #     """
    #     min_std = 0.05
    #     new_std = (
    #         self.max_std
    #         - (self.max_std - min_std) * (timesteps_through / total_timesteps)
    #     ) + 0.0001  # Ensure it never gets to 0, this will give us nans
    #     new_std = 0.20
    #     self.log_std = torch.tensor(np.log(new_std))

    def get_distribution(self, state):
        """Returns a normal distribution, from actions given state

        Args:
            state (torch.Tensor): The current state in environment

        Returns:
            torch.Distribution: A normal distribution, centered around action_values from the network
        """
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
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        noise = torch.randn_like(std)
        probabilities = mean + std * noise
        sampled_action = torch.tanh(probabilities)

        # tanh outputs between -1 and 1, so multiply by action_max to map it to the action space
        scaled_action = sampled_action * self.action_max

        log_2pi = torch.log(torch.Tensor([2 * torch.pi]))
        log_probs = -0.5 * (
            ((probabilities - mean) / std).pow(2) + 2 * log_std + log_2pi
        )

        # one reason for epsilon is to avoid log 0, apparently theres other reasons
        # also idk what this term actually is but they use it in the paper
        epsilon = 1e-6
        log_probs -= torch.log(self.action_max * (1 - sampled_action.pow(2)) + epsilon)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        # could get it to return mean as the 'optimal' action during evaluation?
        return scaled_action, log_probs

    def get_probability_given_action(self, state, action):
        """ """
        scaled_action = action / self.action_max
        scaled_action = torch.clamp(scaled_action, -0.999999, 0.999999)

        pre_tanh = 0.5 * (torch.log(1 + scaled_action) - torch.log(1 - scaled_action))

        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        log_2pi = torch.log(torch.tensor([2 * torch.pi], device=state.device))
        log_probs = -0.5 * (((pre_tanh - mean) / std).pow(2) + 2 * log_std + log_2pi)

        epsilon = 1e-6
        log_probs -= torch.log(self.action_max * (1 - scaled_action.pow(2)) + epsilon)

        log_probs = log_probs.sum(dim=-1, keepdim=False)
        return log_probs

    def evaluate(self, state, action):
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden_values = self.network(state)
        mean = self.mean(hidden_values)
        log_std = self.log_std(hidden_values)
        # they clamp this between -20 and 2 in the paper i believe
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std


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
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state):
        return self.network(state)


class PPOAgent(Agent):
    def __init__(
        self,
        env,
        epsilon=0.2,
        gamma=0.98,
        observation_space=115,  # Default from ant-v4
        action_space=8,
        std=0.4,
        learning_rate=3e-4,
        weight_decay=1e-5,
        lambda_gae=0.95,
        minibatch_size=64,
        num_trajectories=30,  # Note, if this is too high the agent may only run one training loop, so you will not be able to see the change over time. For instance for ant max episode is 1000 timesteps.
        num_epochs=3,
        entropy_coef=0.1,
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
        self.entropy_coef = entropy_coef

    def transfer_policy_net_to_old(self):
        """Copies our current policy into self.old_policy_network"""
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())
        self.old_policy_network.log_std = self.policy_network.log_std

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

        current_log_prob = torch.clamp(current_log_prob, min=-20, max=20)
        old_log_prob = torch.clamp(old_log_prob, min=-20, max=20)
        log_ratio = current_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        ratio = torch.clamp(ratio, min=1e-10, max=1e10)

        return ratio  # This formula is P(A|S) / P_old(A|S) but we can do - since they are log probability

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

    def advantage_estimates_gae(self, states, rewards):
        """
        Use advantage estimation on value network, using GAE
        """
        values = torch.tensor([self.value_network.forward(state) for state in states])
        next_values = values[1:]
        values = values[:-1]
        advantages = torch.zeros_like(rewards)

        rewards = rewards[:-1]
        deltas = rewards + self.gamma * next_values - values

        gae = 0
        for t in reversed(range(len(deltas))):
            gae = (gae * self.gamma * self.lambda_gae) + deltas[t]
            advantages[t] = gae

        return advantages

    def get_trajectories(self):
        trajectories = []

        timesteps_count = 0
        for _ in range(self.num_trajectories):
            is_finished = False
            is_truncated = False
            state, _ = self.env.reset()
            state = torch.tensor(state)
            action, _ = self.policy_network.get_action(
                torch.tensor(state, dtype=torch.float32)
            )
            trajectory = []
            done = False
            # Line 3 of pseudocode, here we are collecting a trajectory
            while not done:
                new_state, reward, is_finished, is_truncated, _ = self.env.step(
                    action.detach().numpy()
                )
                done = is_finished or is_truncated
                # print(is_truncated, reward)
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
        normalisation_factor = 1  # 1 / D_k T, which is just timesteps in trajectory for us because we have 1 trajectory
        network_probability_ratio = (
            torch.stack(  # Stack all the values in our list together,into one big list
                [
                    self.probability_ratios(this_state, this_action)
                    for this_state, this_action in zip(states, actions)
                ]
            )
        )  # pi (a_t| s_t) / pi (a_t, s_t)
        clipped_g = torch.clamp(
            network_probability_ratio, 1 - self.epsilon, 1 + self.epsilon
        )  # g(\epsilon, advantage estimates)
        ratio_with_advantage = advantage_estimates * network_probability_ratio
        clipped_with_advantage = advantage_estimates * clipped_g
        # ppo_clipped = torch.min(ratio_with_advantage, clipped_with_advantage) # TODO original advantage function
        ppo_clipped = torch.min(ratio_with_advantage, clipped_with_advantage)
        policy_loss = -(
            normalisation_factor
            * torch.sum(
                ppo_clipped
            )  # Take the sum of ppo clipped from pseudocode, loops through every timestep/trajectory
        )  # We are minusing here because we are trying to find the arg max, so the LOSS needs to be negative. (since we are trying to minimise the loss)

        # Entropy bonus
        # dist = self.policy_network.get_distribution(states)
        # entropy = dist.entropy().mean()
        # policy_loss = policy_loss - self.entropy_coef * entropy
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
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
        states = torch.stack(
            [this_timestep[0].detach().float() for this_timestep in trajectories], dim=0
        )
        actions = torch.stack(
            [this_timestep[1].detach().float() for this_timestep in trajectories], dim=0
        )
        rewards = torch.tensor([this_timestep[2] for this_timestep in trajectories])

        # Line 4 in pseudocode
        # Compute rewards to go TODO is this the right way to work these out using MC?
        rewards_to_go = self.state_action_values_mc(rewards)
        # Line 5 in Pseudocode
        # Compute advantage estimates
        advantage_estimates = self.advantage_estimates_mc(
            states, rewards
        )  # TODO ONLY NEED ONE OF THESE
        # advantage_estimates = self.advantage_estimates_gae(states, rewards)
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
        timesteps = 0
        episodes = 0
        while timesteps < num_iterations:
            # self.policy_network.update_std(timesteps, num_iterations)
            elapsed_timesteps, reward = (
                self.simulate_episode()
            )  # Simulate an episode and collect rewards
            reward = reward / self.num_trajectories
            timesteps += elapsed_timesteps
            episodes += 1 * self.num_trajectories

            self.reward_list.append(reward)
            self.timestep_list.append(timesteps)

            GLOBAL_TIMESTEPS.append(timesteps)
            GLOBAL_REWARDS.append(reward)

            print(
                f"[Episode {episodes} / timestep {timesteps}] Received reward {reward:.3f}"
            )
        return self.reward_list

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
        # self.policy_network.log_std = torch.tensor(np.log(0.2))
        with torch.no_grad():
            action, _ = self.policy_network.get_action(state)
        return action.detach().numpy()

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


class DPOAgent(PPOAgent):
    def __init__(
        self,
        env,
        epsilon=0.3,
        gamma=0.99,
        observation_space=115,  # Default from ant-v4
        action_space=8,
        std=0.1,
        learning_rate=3e-4,
        weight_decay=0,
        lambda_gae=0.95,
        minibatch_size=512,
        num_trajectories=10,  # Note, if this is too high the agent may only run one training loop, so you will not be able to see the change over time. For instance for ant max episode is 1000 timesteps.
        num_epochs=3,
        entropy_coef=0.01,
        alpha=2,
        beta=0.6,
    ):
        super().__init__(
            env,
            epsilon,
            gamma,
            observation_space,
            action_space,
            std,
            learning_rate,
            weight_decay,
            lambda_gae,
            minibatch_size,
            num_trajectories,
            num_epochs,
            entropy_coef,
        )
        self.alpha = alpha
        self.beta = beta

    def calculate_drift(self, network_probability_ratio, advantage_estimates):
        """
        Drift function (calculating advantage, weighted by how different this is to current policy) from DPO.
        Args:
            network_probability_ratio (torch.Tensor): probabi ratio
            advantage_estimates (torch.Tensor): advantage estimate
        Returns:
            torch.Tensor: Drift value
        """
        drift = torch.zeros_like(advantage_estimates)
        for count, this_estimate in enumerate(
            advantage_estimates
        ):  # Take either the positive or negative drift value
            this_network_probability_ratio = network_probability_ratio[count]
            this_advantage_estimate = advantage_estimates[count]
            if this_estimate > 0:
                drift[count] = torch.nn.functional.relu(
                    (this_network_probability_ratio - 1) * this_advantage_estimate
                    - self.alpha
                    * torch.tanh(
                        ((this_network_probability_ratio - 1) * this_advantage_estimate)
                        / self.alpha
                    )
                )
            else:  # Advantage estimate is negative
                drift[count] = torch.nn.functional.relu(
                    torch.log(this_network_probability_ratio) * this_advantage_estimate
                    - self.beta
                    * torch.tanh(
                        (
                            torch.log(this_network_probability_ratio)
                            * this_advantage_estimate
                        )
                        / self.beta
                    )
                )
        return drift

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

        drift = self.calculate_drift(
            network_probability_ratio, advantage_estimates
        )  # Fig 9 DPO paper, drift is advantage estimate weighted by probability.
        policy_loss = -normalisation_factor * torch.sum(drift * advantage_estimates)

        # Entropy bonus
        dist = self.policy_network.get_distribution(states)
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy

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


def verbose_train(environment):
    """Train our model with progress updates and rendering

    Args:
        environment (array): The environment to train model on, should include name, observation_space, and action_space
    """

    if environment["name"] == "Unitreee":
        env = gym.make(
            "Ant-v5",
            xml_file=r"C:\Users\Solly\_\python\69\Walker\Testing_functions\mujoco_menagerie/unitree_go1/scene.xml",
            forward_reward_weight=1,
            ctrl_cost_weight=0.05,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
            render_mode="rgb_array",
        )
    else:
        env = gym.make(environment["name"], render_mode="rgb_array")
    model = DPOAgent(
        env,
        epsilon=0.2,
        observation_space=environment["observation_space"],
        action_space=environment["action_space"],
        std=0.1,
    )
    model.train(num_iterations=100_000, log_iterations=1000)
    print("\n Training finished")
    print("Rendering...")
    model.render(num_timesteps=100_000)


# environments = [
#     {"name": "InvertedPendulum-v4", "observation_space": 4, "action_space": 1},
#     {"name": "Ant-v4", "observation_space": 27, "action_space": 8},
#     {"name": "Ant-v5", "observation_space": 105, "action_space": 8},
#     {"name": "Unitreee", "observation_space": 115, "action_space": 12},
# ]
# verbose_train(environments[1])
