import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from typing import NamedTuple, Any

# ===========================================================
#
# SAC implementation based on OpenAI Spinning Up pseudocode
# https://spinningup.openai.com/en/latest/algorithms/sac.html
#
# ===========================================================

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
        self.log_std = nn.Linear(self._hidden_units, self._action_dim)

    def forward(self, state:torch.Tensor)->torch.Tensor:
        hidden_values = self.ann(state)
        mean = self.mean(hidden_values)
        log_std = self.log_std(hidden_values)
        # seems like these are standard values to use for clamping
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

        # epsilon is to avoid log 0 and subsequent NaN propagation
        epsilon = 1e-6

        # tanh above squashes distribution to (-1, 1), so we need to adjust log probs accordingly. This is Eq 21 in SAC paper 
        log_probs -= torch.log(self.action_max * (1 - sampled_action.pow(2)) + epsilon)
        log_probs = log_probs.sum(dim=1, keepdim=True)
    
        # could get it to return mean as the 'optimal' action during evaluation?
        return scaled_action, log_probs.squeeze()

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

    def forward(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        # Assuming batch is dim=0, so state is shape [batch_size, state_space]
        # And action is [batch_size, action_space]

        network_input = torch.cat([state, action], dim=1)
        action_value_estimate_1 = self.ann1(network_input) # estimate the value of an action in a state

        return torch.squeeze(action_value_estimate_1)

class SACAgent(Agent):
    def __init__(self, env: gym.Env, update_threshold: int=1, batch_size: int=256, lr: float=3e-4, gamma: float=0.99, polyak=0.995, fixed_alpha=None,reward_scale:float=5):
        super(SACAgent, self).__init__(env)

        # line 1 of pseudocode
        self.env = env
        self.update_threshold = update_threshold
        self.persistent_timesteps = 0
        self.updates = 0

        # model hyperparams
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak

        observation_space_shape = env.observation_space._shape[0]
        action_space_shape = env.action_space._shape[0]
        #we assume action space is symmetrical, as both humanoid and ant are
        action_space_max_value = env.action_space.high[0]
        
        self.actor = SACPolicyNetwork(observation_space_shape,256,action_space_shape,action_space_max_value)
        self.critic_1 = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_2 = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_1_target = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_2_target = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)

        self.actor_loss = SACPolicyLoss()
        self.critic_1_loss = nn.MSELoss()
        self.critic_2_loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(1000000, observation_space_shape, action_space_shape)

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=lr)
        self.critics_optimiser = optim.Adam(list(self.critic_1.parameters())+list(self.critic_2.parameters()), lr=lr)
        
        self.fixed_alpha = fixed_alpha

        if fixed_alpha is None:
            self.log_alpha = torch.tensor(0.0, requires_grad=True)
            self.alpha_optimiser = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -action_space_shape # standard value from hyperparameters appendix of https://arxiv.org/abs/1812.05905
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = fixed_alpha

        #copying critic params into targets
        self.polyak_update(0)

    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)
        for (parameter, target_parameter) in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)

    def train(self, num_timesteps=50000, start_timesteps=1000):
        """Train the agent over a given number of episodes."""
        self.persistent_timesteps = 0
        timesteps = 0

        print(f"Populating replay buffer with {start_timesteps} timesteps of experience...")

        while timesteps < start_timesteps:
            elapsed_timesteps, start_rewards = self.simulate_episode(should_learn=False)
            timesteps += elapsed_timesteps

            self.timestep_list.append(timesteps)
            self.reward_list.append(start_rewards)
                
        super().train(num_timesteps=num_timesteps, start_timesteps=start_timesteps)

    def simulate_episode(self, should_learn=True):
        state, _ = self.env.reset()
        reward_total = 0
        timestep = 0

        # line 3 of pseudocode
        while True:

            timestep += 1
            self.persistent_timesteps += 1

            # line 4 of pseudocode
            if should_learn:
                with torch.no_grad():
                    action = self.actor.sample(torch.tensor([state], dtype=torch.float32))[0].numpy()[0]
            else:
                action = self.env.action_space.sample()

            # line 5-6 of pseudocode
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            reward_total += reward
            reward *= self.reward_scale

            # line 7 of pseudocode
            self.replay_buffer.add(Experience(state, new_state, action, reward, is_finished))
            state = new_state

            # line 9-10 of pseudocode
            if should_learn and self.persistent_timesteps % self.update_threshold == 0:
                self.update_params()

            # line 8 of pseudocode - swapped since we might as well update even if it's a terminal state and the logic is easier this way
            if is_finished or is_truncated:
                break

        return timestep, reward_total

    def update_params(self):
        # line 11 of pseudocode
        old_states, new_states, actions, rewards, terminals = self.replay_buffer.get(self.batch_size)

        old_states = torch.tensor(old_states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)

        critic_1_evaluation = self.critic_1(old_states,actions)
        critic_2_evaluation = self.critic_2(old_states,actions)

        # need this or else temp tuning doesnt work, also speeds it up
        with torch.no_grad():
            # line 12 of pseudocode
            actor_prediction_new, log_actor_probability_new = self.actor.sample(new_states)

            critic_target_1_prediction = self.critic_1_target(new_states, actor_prediction_new)
            critic_target_2_prediction = self.critic_2_target(new_states, actor_prediction_new)
            critic_target_clipped = torch.min(critic_target_1_prediction, critic_target_2_prediction)

            predicted_target_reward = (critic_target_clipped - self.alpha * log_actor_probability_new)
            target = rewards + self.gamma * (1 - terminals) * predicted_target_reward
        
        # line 13 of pseudocode
        critic_1_loss = self.critic_1_loss(critic_1_evaluation,target)
        critic_2_loss = self.critic_2_loss(critic_2_evaluation,target)
        total_critic_loss = critic_1_loss + critic_2_loss

        self.critics_optimiser.zero_grad()
        total_critic_loss.backward()
        self.critics_optimiser.step()

        # line 14 of pseudocode
        actor_prediction_old, log_actor_probability_old = self.actor.sample(old_states)
        critic_1_prediction = self.critic_1(old_states, actor_prediction_old)
        critic_2_prediction = self.critic_2(old_states, actor_prediction_old)
        critic_clipped = torch.min(critic_1_prediction, critic_2_prediction)

        actor_loss = self.actor_loss(critic_clipped,log_actor_probability_old,self.alpha)

        if self.fixed_alpha is None:
            alpha_loss = -(self.log_alpha * (log_actor_probability_old + self.target_entropy).detach()).mean()
        else:
            alpha_loss = None

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        if self.fixed_alpha is None:
            self.alpha_optimiser.zero_grad()
            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

        # line 15 of pseudocode
        self.polyak_update(self.polyak)
    
    def predict(self, state):
        with torch.no_grad():
            action = self.actor.forward(state.unsqueeze(0))
            scaled_action = torch.tanh(action[0])*self.actor.action_max
        return scaled_action.detach().numpy()[0]


if __name__ == "__main__":
    env = gym.make("Ant-v4", render_mode="rgb_array")
    agent = SACAgent(env,fixed_alpha=0.2,reward_scale=5.0,lr=3e-4,batch_size=256)
    agent.train(num_timesteps=1_000_000,start_timesteps=10_000)

