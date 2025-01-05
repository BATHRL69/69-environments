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
            nn.Linear(256, action_space),
        )
        self.max_std = std
        self.log_std = torch.tensor(np.log(std))

    def update_std(self, timesteps_through, total_timesteps):
        """Update STD over training, to encourage exploration at the start, and then hone in on correct actions neared the end

        Args:
            timesteps_through (int): Amount of timesteps through at curent point in training
            total_timesteps (int): Total timesteps for training
        """
        min_std = 0.05
        new_std = (
            self.max_std
            - (self.max_std - min_std) * (timesteps_through / total_timesteps)
        ) + 0.01  # Ensure it never gets to 0, timesteps through can go slightly above timesteps, this will give us nans
        self.log_std = torch.tensor(np.log(new_std))

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
        distribution = self.get_distribution(state)
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
        distribution = self.get_distribution(state)
        log_probs = distribution.log_prob(action).sum(dim=-1)

        return log_probs  # We can do sum here because they are LOG probs, usually our conditional probability would be x * y.
        # We are doing exp to turn our log_prob into probability 0-1. Will do torch.exp in probability ratio method to return this to between 0 and 1

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
            nn.Linear(observation_space, 256),
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
        gamma=0.99,
        observation_space=115,  # Default from ant-v4
        action_space=8,
        std=0.4,
        learning_rate=3e-4,
        weight_decay=1e-5,
        lambda_gae=0.95,
        batch_size=64,
        num_trajectories=10,  # Note, if this is too high the agent may only run one training loop, so you will not be able to see the change over time. For instance for ant max episode is 1000 timesteps.
        num_epochs=3,
        entropy_coef=0.01,
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
        self.minibatch_size = batch_size
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
        ratio = torch.exp(current_log_prob - old_log_prob)

        ratio = torch.clamp(
            ratio, min=1e-2, max=1e2
        )  # Avoid a huge change if we are doing lots of timesteps between changes to networks

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
        state_value_estimates = self.value_network(states).squeeze(-1)  # V(S)
        state_action_value_estimates = self.state_action_values_mc(rewards)  # Q(S,A)
        return (
            state_action_value_estimates.float() - state_value_estimates
        )  # Q(S,A) - V(S)

    def advantage_estimates_gae(self, states, rewards):
        """
        Use advantage estimation on value network, using GAE
        Args:
            states (torch.Tensor) :
            rewards (torch.Tensor) :
        Returns:
            torch.Tensor: The advantage estimate for each step in trajectory.
        """
        values = torch.tensor([self.value_network.forward(state) for state in states])
        next_values = values[1:]
        values = values[:-1]
        advantages = torch.zeros_like(rewards)

        rewards = rewards[:-1]  # Final reward is for a state that will not be calculated
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

            trajectories.append(trajectory)
        return trajectories, timesteps_count

    def update_params(
        self, rewards_to_go, advantage_estimates, states, actions, total_timesteps
    ):
        # Line 6 in Pseudocode
        normalisation_factor = (
            1 / total_timesteps
        )  # 1 / D_k T, which is just timesteps in trajectory for us because we have 1 trajectory
        network_probability_ratio = (
            torch.stack(  # Stack all the values in our list together,into one big list
                [
                    self.probability_ratios(this_state, this_action)
                    for this_state, this_action in zip(states, actions)
                ]
            )
        ).squeeze()  # pi (a_t| s_t) / pi (a_t, s_t)
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

        # Entropy bonus
        dist = self.policy_network.get_distribution(states)
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), max_norm=1.0
        )  # Avoid huge grad updates
        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()
        # Line 7 in pseudocode
        value_estimates = self.value_network(states).squeeze()
        self.value_optimiser.zero_grad()
        value_loss = torch.mean(torch.square(value_estimates - rewards_to_go))
        value_loss.backward()
        self.value_optimiser.step()

    def simulate_episode(self):
        """Simulate a single episode, called by train method on parent class"""
        self.transfer_policy_net_to_old()  # Transfer the current probability network to OLD, so that we are feezing it, before we start making updates

        trajectories, total_timesteps = self.get_trajectories()
        all_states = []
        all_actions = []
        all_rewards_to_go = []
        all_advantage_estimates = []
        all_rewards = []
        for this_trajectory in trajectories:
            states = torch.stack(
                [
                    this_timestep[0].detach().float()
                    for this_timestep in this_trajectory
                ],
                dim=0,
            )
            actions = torch.stack(
                [
                    this_timestep[1].detach().float()
                    for this_timestep in this_trajectory
                ],
                dim=0,
            )
            rewards = torch.tensor(
                [this_timestep[2] for this_timestep in this_trajectory]
            )
            # Line 4 in pseudocode
            rewards_to_go = self.state_action_values_mc(rewards)
            # Line 5 in Pseudocode
            # Compute advantage estimates
            advantage_estimates = self.advantage_estimates_gae(states, rewards)
            all_rewards.append(rewards)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards_to_go.append(rewards_to_go)
            all_advantage_estimates.append(advantage_estimates)
        all_rewards = torch.cat(all_rewards, dim=0)
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_rewards_to_go = torch.cat(all_rewards_to_go, dim=0)
        all_advantage_estimates = torch.cat(all_advantage_estimates, dim=0)
        for _ in range(self.num_epochs):
            batch_locations = np.arange(total_timesteps)
            np.random.shuffle(batch_locations)

            for this_batch_start in range(0, total_timesteps, self.minibatch_size):
                this_batch_end = this_batch_start + self.minibatch_size
                this_batch = batch_locations[this_batch_start:this_batch_end]
                length_of_batch = len(this_batch)
                self.update_params(
                    all_rewards_to_go[this_batch],
                    all_advantage_estimates[this_batch],
                    all_states[this_batch],
                    all_actions[this_batch],
                    length_of_batch,
                )
        return total_timesteps, torch.sum(all_rewards)

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
            self.policy_network.update_std(timesteps, num_iterations)
            elapsed_timesteps, reward = (
                self.simulate_episode()
            )  # Simulate an episode and collect rewards
            reward = reward / self.num_trajectories
            timesteps += elapsed_timesteps
            episodes += self.num_trajectories

            self.reward_list.append(reward)
            self.timestep_list.append(timesteps)

            GLOBAL_TIMESTEPS.append(timesteps)
            GLOBAL_REWARDS.append(reward)

            print(
                f"[Episode {episodes} / timestep {timesteps}] Received reward {reward:.3f}"
            )
        return self.reward_list, self.timestep_list

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
        self.policy_network.log_std = torch.tensor(
            np.log(0.05)
        )  # Some stochasicity in video
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
        epsilon=0.2,
        gamma=0.99,
        observation_space=115,  # Default from ant-v4
        action_space=8,
        std=0.4,
        learning_rate=3e-4,
        weight_decay=1e-5,
        lambda_gae=0.95,
        batch_size=64,
        num_trajectories=10,  # Note, if this is too high the agent may only run one training loop, so you will not be able to see the change over time. For instance for ant max episode is 1000 timesteps.
        num_epochs=3,
        entropy_coef=0.01,
        alpha=2.5,
        beta=0.7,
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
            batch_size,
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
    
        network_probability_ratio = (
            torch.stack(  # Stack all the values in our list together,into one big list
                [
                    self.probability_ratios(this_state, this_action)
                    for this_state, this_action in zip(states, actions)
                ]
            )
        )  # pi (a_t| s_t) / pi (a_t, s_t)

        drift = self.calculate_drift(
            network_probability_ratio, advantage_estimates
        )  # Fig 9 DPO paper, drift is advantage estimate weighted by probability.
        policy_loss = -(network_probability_ratio * advantage_estimates - drift).mean()

        # Entropy bonus
        dist = self.policy_network.get_distribution(states)
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), max_norm=1.0
        )  # Avoid huge grad updates
        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()

        # Line 7 in pseudocode
        value_estimates = self.value_network(states).squeeze()
        self.value_optimiser.zero_grad()

        value_loss = torch.mean(torch.square(value_estimates - rewards_to_go))
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


environments = [
    {"name": "InvertedPendulum-v4", "observation_space": 4, "action_space": 1},
    {"name": "Ant-v4", "observation_space": 27, "action_space": 8},
    {"name": "Ant-v5", "observation_space": 105, "action_space": 8},
    {"name": "Unitreee", "observation_space": 115, "action_space": 12},
]
# Uncomment me to train
# verbose_train(environments[1])
