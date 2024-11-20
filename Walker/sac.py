import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from typing import NamedTuple, Any

# Experience as a named tuple is not torch friendly ( cant parallelise using indexing ) and therefore we are looking for a way around it
# Problem is these arrays must be homogenous in shape, but since a state is not the same size as e.g. is_terminal 
# we will have to seperate them in the replay buffer, or have them as one numpy array

class Experience(NamedTuple):
    old_state: Any
    new_state: Any
    action: Any
    reward: float
    is_terminal: bool

# Experience is going to be a numpy array as such: Experience = [[old_state],[new_state],[action],[reward],[is_terminal]]
# or [old_state,new_state,action,reward,is_terminal]

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
        indices = np.random.sample(list(range(valid_entries)), batch_size)
        return self.old_state_buffer[indices], self.new_state_buffer[indices], self.action_buffer[indices], self.reward_buffer[indices], self.is_terminal_buffer[indices]


class SACPolicyLoss(nn.Module):
    def __init__(self):
        super(SACPolicyLoss, self).__init__()
    
    def forward(self, min_critic, entropy, alpha):
        return min_critic - alpha * entropy


class SACPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int=256, hidden_units:int=256, action_dim: int=1):
        super(SACPolicyNetwork, self).__init__()

        self._input_dim = input_dim
        self._action_dim = action_dim
        self._hidden_units = hidden_units

        self.ann = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_units),
            nn.ReLU(),
            nn.Linear(self._hidden_units, self._hidden_units),
            nn.Sigmoid(), # might have to change this
        )

        self.mean = nn.Linear(self._hidden_units, self._action_dim)
        self.log_std_dev = nn.Linear(self._hidden_units, self._action_dim)

    def forward(self, state:torch.Tensor, train:bool=True)->torch.Tensor:
        #state.unsqueeze(0)
        out = self.ann(state) # samples a prob dis
        mean = self.mean(out)
        ## why log std and why not std
        log_std = self.log_std_dev(out)
        log_std = torch.clamp(log_std,min=-20,max=2) ### these values are used by seemingly everyone, although we do not yet know why or where to reference 
        return mean, log_std
    
    def sample(self, state:torch.Tensor)->torch.Tensor:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        noise = torch.randn_like(std)
        sampled_action = mean + std * noise
        log_2_pi = torch.log(torch.Tensor([2 * torch.pi]))
        log_prob = -0.5 * ((noise**2) + 2 * log_std + log_2_pi).sum(dim=1, keepdim=True) # do we need dim=1 and keepdim
        return sampled_action, log_prob


class SACValueNetwork(nn.Module):
    def __init__(self, input_dim:int=256, hidden_dim=256):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        super(SACValueNetwork, self).__init__()
        self.ann = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, 1),
        )

    def forward(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        # Assuming batch is dim=0, so state is shape [batch,state_space]
        # action is [batch,action_space]

        network_input = torch.cat([state,action],dim=1)
        action_value_estimate = self.ann(network_input) # estimate the value of an action in a state
        return action_value_estimate


class SACAgent(Agent):
    # TODO: count timesteps across episodes, not within - agent still needs to learn on short episodes

    def __init__(self, env: gym.Env, update_threshold: int = 100, batch_size: int = 256, alpha: float = 1e-4, gamma: float = 0.99, polyak = 0.99):
        # line 1 of pseudocode
        self.env = env
        self.update_threshold = update_threshold
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak

        observation_space_shape = env.observation_space._shape[0]
        action_space_shape = env.action_space._shape[0]

        self.replay_buffer = ReplayBuffer(1000000, observation_space_shape, action_space_shape)
        self.actor = SACPolicyNetwork(input_dim=observation_space_shape, action_dim=action_space_shape)
        self.critic_1 = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_2 = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_target_1 = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
        self.critic_target_2 = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)

        self.actor_loss = SACPolicyLoss()
        self.critic_1_loss = nn.MSELoss()
        self.critic_2_loss = nn.MSELoss()

        self.actor_optimiser = optim.Adam(self.actor.parameters())
        self.critic_1_optimiser = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimiser = optim.Adam(self.critic_2.parameters())

        # line 2 of pseudocode
        self.polyak_update(0)


    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)
        
        for (parameter, target_parameter) in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)

    
    def simulate_episode(self, skip_update=False):
        is_finished = False
        is_truncated = False

        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        timestep = 0

        # line 3 of pseudocode
        while True:
            timestep += 1

            # line 4 of pseudocode
            action = self.actor.sample(state)[0].detach().numpy()

            # line 5-6 of pseudocode
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action[0])
            new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)

            # line 7 of pseudocode
            self.replay_buffer.add(Experience(state[0], new_state[0], action, reward, is_finished))

            # line 8 of pseudocode
            if is_finished or is_truncated:
                break

            # line 9 of pseudocode
            if skip_update or timestep % self.update_threshold != 0:
                continue

            # line 11 of pseudocode
            old_states, new_states, actions, rewards, terminals = self.replay_buffer.get(self.batch_size)

            # line 12 of pseudocode
            actor_prediction_new, log_actor_probability_new = self.actor.forward(new_states, train=False)
            critic_target_1_prediction = self.critic_target_1.forward(new_states, actor_prediction_new, train=False)
            critic_target_2_prediction = self.critic_target_2.forward(new_states, actor_prediction_new, train=False)
            critic_target_clipped = min(critic_target_1_prediction, critic_target_2_prediction)
            predicted_target_reward = critic_target_clipped - self.alpha * log_actor_probability_new
            target = rewards + self.gamma * (1 - terminals) * predicted_target_reward

            # line 13 of pseudocode
            # TODO: batch norm
            critic_1_evaluation = self.critic_1.forward(old_states, actions, train=False)
            critic_2_evaluation = self.critic_2.forward(old_states, actions, train=False)

            critic_1_loss = self.critic_1_loss(target, critic_1_evaluation)
            critic_2_loss = self.critic_2_loss(target, critic_2_evaluation)

            self.critic_1_optimiser.zero_grad()
            self.critic_2_optimiser.zero_grad()

            critic_1_loss.backward()
            critic_2_loss.backward()

            self.critic_1_optimiser.step()
            self.critic_2_optimiser.step()

            # line 14 of pseudocode
            # TODO: batch norm
            actor_prediction_old, log_actor_probability_old = self.actor.forward(old_states, train=False)
            critic_1_prediction = self.critic_1.forward(old_states, actor_prediction_old, train=False)
            critic_2_prediction = self.critic_2.forward(old_states, actor_prediction_old, train=False)
            critic_clipped = min(critic_1_prediction, critic_2_prediction)

            actor_loss = self.actor_loss(critic_clipped, log_actor_probability_old, self.alpha)

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_optimiser.step()

            # line 15 of pseudocode
            self.polyak_update(self.polyak)
            
            state = new_state

        return timestep


    def train(self, num_timesteps=100000, print_interval=50, start_timesteps=10000):
        """Train the agent over a given number of episodes."""
        timesteps = 0
        episodes = 0

        while timesteps < start_timesteps:
            timesteps += self.simulate_episode(skip_update=True)
            episodes += 1

            if (episodes % print_interval == 0):
                print(f"Start timesteps {100 * timesteps / start_timesteps:.2f}% complete...")
        
        super().train(num_timesteps - timesteps, print_interval)



    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError





env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")

model = SACAgent(env)
model.train(num_timesteps=100000)