from copy import deepcopy
import cv2
import gymnasium as gym
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from typing import NamedTuple, Any


class Experience(NamedTuple):
    old_state: Any
    new_state: Any
    action: Any
    reward: float
    is_terminal: bool


class ReplayBuffer():
    def __init__(self, max_capacity: int, state_shape_size: int, action_space_size: int):
        self.max_capacity = max_capacity
        self.counter = 0

        self.old_state_buffer = np.zeros((max_capacity, state_shape_size))
        self.new_state_buffer = np.zeros((max_capacity, state_shape_size))
        self.action_buffer = np.zeros((max_capacity, action_space_size))
        self.reward_buffer = np.zeros(max_capacity)
        self.is_terminal_buffer = np.zeros(max_capacity)     
    

    def add(self, experience: Experience):
        idx = self.counter % self.max_capacity
        self.counter += 1

        self.old_state_buffer[idx] = experience.old_state
        self.new_state_buffer[idx] = experience.new_state
        self.action_buffer[idx] = experience.action
        self.reward_buffer[idx] = experience.reward
        self.is_terminal_buffer[idx] = experience.is_terminal
    

    def get(self, batch_size: int):
        valid_entries = min(self.counter, self.max_capacity)
        indices = np.random.choice(list(range(valid_entries)), batch_size)
        return self.old_state_buffer[indices], self.new_state_buffer[indices], self.action_buffer[indices], self.reward_buffer[indices], self.is_terminal_buffer[indices]


class SACPolicyLoss(nn.Module):
    def __init__(self):
        super(SACPolicyLoss, self).__init__()
    

    def forward(self, min_critic, entropy, alpha):
        # gradient ASCENT - negate the loss function in the pseudocode, since our optimiser will perform gradient DESCENT
        return torch.mean(alpha * entropy - min_critic)


class SACPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int=256, hidden_units: int=256, action_dim: int=1, action_max:float=1):
        super(SACPolicyNetwork, self).__init__()

        self._input_dim = input_dim
        self._action_dim = action_dim
        self._hidden_units = hidden_units
        self.action_max = action_max

        self.ann = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_units),
            nn.ReLU(),
            nn.Linear(self._hidden_units, self._hidden_units),
            nn.ReLU()
        )

        self.mean = nn.Linear(self._hidden_units, self._action_dim)
        # attempting to simplify this to std for now
        # note: attempt didnt work because: log negative numbers = bad
        self.log_std = nn.Linear(self._hidden_units, self._action_dim)


    def forward(self, state:torch.Tensor)->torch.Tensor:
        hidden_values = self.ann(state)
        mean = self.mean(hidden_values)
        log_std = self.log_std(hidden_values)
        # they clamp this between -20 and 2 in the paper i believe
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std
    

    def sample(self, state:torch.Tensor)->torch.Tensor:
        ## One issue that was encountered was when trying to do just the std
        ## as the output, and then logging it without clamping it to appropriate values
        ## so NAN values would appear due to log 0
        ## the same thing happened if the term (1 - sampled_action.pow(2) wasn't scaled correctly)
        ## i.e. if you do 1 - scaled_action.pow(2), since then scaled_action could be >1 
        ## so you'd get log(negative)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        noise = torch.randn_like(std)    
        probabilities = mean + std * noise
        sampled_action = torch.tanh(probabilities)

        # tanh outputs between -1 and 1, so multiply by action_max to map it to the action space
        scaled_action = sampled_action * self.action_max

        log_2pi = torch.log(torch.Tensor([2 * torch.pi]))
        log_probs = -0.5 * (((probabilities - mean) / std).pow(2) + 2 * log_std + log_2pi)

        # one reason for epsilon is to avoid log 0, apparently theres other reasons
        # also idk what this term actually is but they use it in the paper
        epsilon = 1e-6
        log_probs -= torch.log(self.action_max * (1 - sampled_action.pow(2)) + epsilon)
        log_probs = log_probs.sum(dim=1, keepdim=True)
    
        # could get it to return mean as the 'optimal' action during evaluation?
        return scaled_action, log_probs


class SACValueNetwork(nn.Module):
    def __init__(self, input_dim:int=256, hidden_dim:int=256):
        super(SACValueNetwork, self).__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        # they used 2 hidden layers and 256 hidden units in paper
        self.ann1 = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, 1),
        )

        self.ann2 = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, 1),
        )


    def forward(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        # Assuming batch is dim=0, so state is shape [batch,state_space]
        # action is [batch,action_space]

        network_input = torch.cat([state, action], dim=1)
        action_value_estimate_1 = self.ann1(network_input) # estimate the value of an action in a state
        action_value_estimate_2 = self.ann2(network_input) # estimate the value of an action in a state

        return action_value_estimate_1, action_value_estimate_2


## WORKING AGENT
class SACAgent(Agent):
    def __init__(self, env: gym.Env, update_threshold: int=1, batch_size: int=256, lr: float=3e-4, gamma: float=0.99, polyak=0.995, fixed_alpha=None):
        super(SACAgent, self).__init__(env)

        # line 1 of pseudocode
        self.env = env
        self.update_threshold = update_threshold
        self.persistent_timesteps = 0
        self.updates = 0

        # model hyperparams
        self.batch_size = batch_size
        self.fixed_alpha = fixed_alpha
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak

        observation_space_shape = env.observation_space._shape[0]
        action_space_shape = env.action_space._shape[0]
        action_space_max_value = env.action_space.high[0]

        self.replay_buffer = ReplayBuffer(1000000, observation_space_shape, action_space_shape)
        
        self.actor = SACPolicyNetwork(input_dim=observation_space_shape, action_dim=action_space_shape, action_max=action_space_max_value)
        self.critics = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_targets = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)

        self.actor_loss = SACPolicyLoss()
        self.critic_1_loss = nn.MSELoss()
        self.critic_2_loss = nn.MSELoss()

        self.target_entropy = -action_space_shape
        self.log_alpha = torch.tensor(0.0, requires_grad=True)

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critics_optimiser = optim.Adam(self.critics.parameters(), lr=self.lr)
        self.alpha_optimiser = optim.Adam([self.log_alpha], lr=self.lr)

        # line 2 of pseudocode
        self.polyak_update(0)


    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critics.parameters(), self.critic_targets.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)


    def train(self, num_timesteps=50000, start_timesteps=1000):
        """Train the agent over a given number of episodes."""
        self.persistent_timesteps = 0
        timesteps = 0

        self.reward_list.append(0)
        self.timestep_list.append(0)

        print(f"Populating replay buffer with {start_timesteps} timesteps of experience...")

        while timesteps < start_timesteps:
            elapsed_timesteps, _ = self.simulate_episode(should_learn=False)
            timesteps += elapsed_timesteps
                
        super().train(num_timesteps=num_timesteps, start_timesteps=start_timesteps)


    def simulate_episode(self, should_learn=True):
        reward_total = 0

        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        timestep = 0

        # line 3 of pseudocode
        while True:
            timestep += 1
            self.persistent_timesteps += 1

            # line 4 of pseudocode
            if should_learn:
                action = self.actor.sample(state)[0].detach().numpy()
                new_state, reward, is_finished, is_truncated, _ = self.env.step(action[0])
            else:
                action = self.env.action_space.sample()
                new_state, reward, is_finished, is_truncated, _ = self.env.step(action)

            reward_total += reward

            # line 5-6 of pseudocode
            new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)

            # line 7 of pseudocode
            self.replay_buffer.add(Experience(state.flatten(), new_state.flatten(), action, reward, is_finished))

            # line 8 of pseudocode
            if is_finished or is_truncated:
                break

            # line 9 of pseudocode
            if not should_learn or self.persistent_timesteps % self.update_threshold != 0:
                continue

            # lines 11-15 of pseudocode
            self.update_params()
            
            state = new_state

        return timestep, reward_total
    

    def update_params(self):
        # line 11 of pseudocode
        old_states, new_states, actions, rewards, terminals = self.replay_buffer.get(self.batch_size)

        old_states = torch.tensor(old_states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        terminals = torch.tensor(terminals, dtype=torch.float32).unsqueeze(1)

        # line 12 of pseudocode
        actor_prediction_new, log_actor_probability_new = self.actor.sample(new_states)

        critic_target_1_prediction, critic_target_2_prediction = self.critic_targets.forward(new_states, actor_prediction_new)
        critic_target_clipped = torch.min(critic_target_1_prediction, critic_target_2_prediction)

        alpha = self.log_alpha.exp() if self.fixed_alpha is None else self.fixed_alpha

        predicted_target_reward = critic_target_clipped - alpha * log_actor_probability_new
        target = rewards + self.gamma * (1 - terminals) * predicted_target_reward

        # line 13 of pseudocode
        critic_1_evaluation, critic_2_evaluation = self.critics.forward(old_states, actions)

        critic_1_loss = self.critic_1_loss(critic_1_evaluation, target)
        critic_2_loss = self.critic_2_loss(critic_2_evaluation, target)
        total_critic_loss = critic_1_loss + critic_2_loss

        self.critics_optimiser.zero_grad()
        total_critic_loss.backward(retain_graph=True)
        self.critics_optimiser.step()

        # line 14 of pseudocode
        actor_prediction_old, log_actor_probability_old = self.actor.sample(old_states)

        critic_1_prediction, critic_2_prediction = self.critics.forward(old_states, actor_prediction_old)
        critic_clipped = torch.min(critic_1_prediction, critic_2_prediction)

        actor_loss = self.actor_loss(critic_clipped, log_actor_probability_old, alpha)

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # temperature tuning
        alpha_loss = -(self.log_alpha * (log_actor_probability_old + self.target_entropy).detach()).mean()

        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()

        self.polyak_update(self.polyak)


    def predict(self, state):
        
        with torch.no_grad():
            action = self.actor.forward(state.unsqueeze(0))
            scaled_action = torch.tanh(action[0])*self.actor.action_max
        return scaled_action.detach().numpy()[0]
        # return action[0].detach().numpy()[0]
    

    def save(self, path):
        print(f"Saving model to {path}...")
        with open(path, "wb") as file:
            pickle.dump((self.actor.state_dict(), 
                         self.critics.state_dict(), 
                         self.critic_targets.state_dict(), 
                         self.actor_optimiser.state_dict(),
                         self.critics_optimiser.state_dict(),
                         self.replay_buffer), file)

    
    def load(self, path):
        if os.path.exists(path):
            print(f"Loading model from {path}...")
            with open(path, "rb") as file:
                actor_dict, critics_dict, critic_targets_dict, actor_optim_dict, critics_optim_dict, self.replay_buffer = pickle.load(file)
                self.actor.load_state_dict(actor_dict)
                self.critics.load_state_dict(critics_dict)
                self.critic_targets.load_state_dict(critic_targets_dict)
                self.actor_optimiser.load_state_dict(actor_optim_dict)
                self.critics_optimiser.load_state_dict(critics_optim_dict)


# env = gym.make("Ant-v4", render_mode="rgb_array")
# SAVE_PATH = "sac_ant2.data"

# train_agent = SACAgent(env)
# train_agent.load(SAVE_PATH)
# agent.train(num_timesteps=2_000, start_timesteps=1000)
# agent.save(SAVE_PATH)
# train_agent.render()

# obs, info = env.reset()

# for i in range(10_000):
#     action = train_agent.actor.sample(torch.Tensor([obs]))
#     obs, reward, done, trunacted ,info = env.step(action[0].detach().numpy()[0])
#     img = env.render()
#     print(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Double Inverted Pendulum", img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     if done:
#         obs, info = env.reset()

# env.close()

# env = gym.make("Humanoid-v4", render_mode="rgb_array")
# SAVE_PATH = "sac_humanoid.data"

# agent = SACAgent(env)
# agent.load(SAVE_PATH)
# agent.train(num_timesteps=100000, start_timesteps=0)
# agent.save(SAVE_PATH)
# agent.render()

# env.close()