import itertools
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from agent import Agent
from typing import NamedTuple, Any


def polyak_update(critics,critic_targets,polyak):
    for (parameter, target_parameter) in zip(critics.parameters(), critic_targets.parameters()):
        target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)


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

    def __len__(self):
        return self.counter

## WORKING AGENT
class SACAgent(Agent):

    def __init__(self, env: gym.Env, update_threshold: int = 1, batch_size: int = 256, lr: float = 3e-4, gamma: float = 0.99, polyak = 0.995,alpha:float = 0.2):
        # line 1 of pseudocode
        self.loss_history = [0]
        self.loss_history_actor = [0]
        self.some_values = []
        self.last_printed = 0.1

        self.env = env
        self.persistent_timesteps = 0

        self.updates=0

        # model hyperparams
        self.batch_size = batch_size
        self.alpha = alpha
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak

        observation_space_shape = env.observation_space._shape[0]
        action_space_shape = env.action_space._shape[0]
        action_space_max_value = env.action_space.high[0]

        #replay buffer inside bad?!?!
        self.replay_buffer = ReplayBuffer(1000000, observation_space_shape, action_space_shape)
        
        self.actor = SACPolicyNetwork(input_dim=observation_space_shape, action_dim=action_space_shape,action_max=action_space_max_value)

        self.critics = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)

        self.critic_targets = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)


        self.actor_loss = SACPolicyLoss()
        self.critic_1_loss = nn.MSELoss()
        self.critic_2_loss = nn.MSELoss()

        self.actor_optimiser = optim.Adam(self.actor.parameters(),lr=self.lr)

        self.critics_optimiser = optim.Adam(self.critics.parameters(),lr=self.lr)

        # line 2 of pseudocode
        self.polyak_update(0)

    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critics.parameters(), self.critic_targets.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)
    
    def update_params(self,replay_buffer):
        # line 11 of pseudocode
            old_states, new_states, actions, rewards, terminals = replay_buffer.get(self.batch_size)

            old_states = torch.tensor(old_states, dtype=torch.float32)
            new_states = torch.tensor(new_states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            terminals = torch.tensor(terminals, dtype=torch.float32).unsqueeze(1)

            # line 12 of pseudocode
            with torch.no_grad():
                actor_prediction_new, log_actor_probability_new = self.actor.sample(new_states)

                critic_target_1_prediction, critic_target_2_prediction = self.critic_targets.forward(new_states,actor_prediction_new)
                critic_target_clipped = torch.min(critic_target_1_prediction, critic_target_2_prediction)

                predicted_target_reward = critic_target_clipped - self.alpha * log_actor_probability_new
                target = rewards + self.gamma * terminals * predicted_target_reward

            # line 13 of pseudocode
            critic_1_evaluation, critic_2_evaluation = self.critics.forward(old_states, actions)

            critic_1_loss = self.critic_1_loss(critic_1_evaluation, target)
            critic_2_loss = self.critic_2_loss(critic_2_evaluation, target)
            total_critic_loss = critic_1_loss + critic_2_loss

            self.critics_optimiser.zero_grad()

            total_critic_loss.backward()
            
            self.critics_optimiser.step()

            # line 14 of pseudocode
            actor_prediction_old, log_actor_probability_old = self.actor.sample(old_states)
            critic_1_prediction, critic_2_prediction = self.critics.forward(old_states, actor_prediction_old)
            critic_clipped = torch.min(critic_1_prediction, critic_2_prediction)

            actor_loss = self.actor_loss(critic_clipped, log_actor_probability_old, self.alpha)


            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_optimiser.step()

            ############ SOME VERY BAD LOGGING
            
            detached_loss = total_critic_loss.detach().numpy()

            if detached_loss != self.loss_history[-1]:
                self.loss_history.append(detached_loss)

            detached_actor_loss = actor_loss.detach().numpy()

            if detached_actor_loss != self.loss_history_actor[-1]:
                self.loss_history_actor.append(detached_actor_loss)

            polyak_update(self.critics,self.critic_targets,self.polyak)

    def train(self,replay_buffer=None, num_timesteps=100000, print_interval=500, start_timesteps=10000):
        """Train the agent over a given number of episodes."""
        self.persistent_timesteps = 0
        timesteps = 0
        episodes = 0

        while timesteps < start_timesteps:
            elapsed_timesteps, _ , _, _ = self.simulate_episode(replay_buffer,skip_update=True)
            timesteps += elapsed_timesteps
            episodes += 1

            if (episodes % print_interval == 0):
                print(f"Start timesteps {100 * timesteps / start_timesteps:.2f}% complete...")
        
        super().train(replay_buffer,num_timesteps - timesteps, print_interval)


    def simulate_episode(self,replay_buffer,skip_update=False):
        is_finished = False
        is_truncated = False
        reward_history = []

        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        timestep = 0

        # line 3 of pseudocode
        while True:
            timestep += 1
            self.persistent_timesteps += 1

            # line 4 of pseudocode
            if skip_update:
                action = self.env.action_space.sample()
                new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            else:
                action = self.actor.sample(state)[0].detach().numpy()
                new_state, reward, is_finished, is_truncated, _ = self.env.step(action[0])

            # line 5-6 of pseudocode

            # new_state, reward, is_finished, is_truncated, _ = self.env.step(action[0])

            reward = reward # can scale

            new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
            reward_history.append(reward)

            # line 7 of pseudocode
            replay_buffer.add(Experience(state[0], new_state[0], action, reward, is_finished))

            # line 8 of pseudocode
            if is_finished or is_truncated:
                break

            # line 9 of pseudocode
            if skip_update or self.persistent_timesteps % self.update_threshold != 0:
                continue

            # lines 11-15 of pseudocode
            self.update_params(replay_buffer)
            
            state = new_state
        
        if self.loss_history[-1] != self.last_printed:
            self.last_printed = self.loss_history[-1]
            print(self.loss_history[-1])

        return timestep, sum(reward_history), self.loss_history, self.loss_history_actor

def initialize_weights_xavier(m):
    if isinstance(m, nn.Linear):  
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  
            torch.nn.init.constant_(m.bias, 0)

class SACPolicyLoss(nn.Module):
    def __init__(self):
        super(SACPolicyLoss, self).__init__()
    
    def forward(self, min_critic, entropy, alpha):
        # This way works
        return torch.mean(alpha*entropy - min_critic)
        # This way correct
        # return torch.mean(min_critic - alpha*entropy)


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
        self.apply(initialize_weights_xavier)

    def forward(self, state:torch.Tensor)->torch.Tensor:
        out = self.ann(state) # samples a prob dis??? idk where this statement came from 

        mean = self.mean(out)

        log_std = self.log_std(out)
        # they clamp this between -20 and 2 in the paper i believe
        log_std = torch.clamp(log_std,min=-20,max=2)

        return mean, log_std
    
    def sample(self, state:torch.Tensor)->torch.Tensor:
        ## One issue that was encountered was when trying to do just the std
        ## as the output, and then logging it without clamping it to appropriate values
        ## so NAN values would appear due to log 0
        ## the same thing happened if the term (1-sampled_action.pow(2) wasn't scaled correctly)
        ## i.e. if you do 1- scaled_action.pow(2), since then scaled_action could be >1 
        # so you'd get log(negative)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        noise = torch.randn_like(std)    
        
        probabilities = mean + std * noise
        sampled_action = torch.tanh(probabilities)
         # tanh between -1 and 1 so we times by action_max to map it to action space
        scaled_action = sampled_action * self.action_max

        log_2pi = torch.log(torch.Tensor([2 * torch.pi]))
        log_probs = -0.5 * (((probabilities - mean) / std).pow(2) + 2 * log_std + log_2pi)

        # one reason for epsilon (1e-6) is to avoid log 0, apparently theres other reasons
        # also idk what this term actually is but they use it in the paper
        log_probs -= torch.log(self.action_max * (1 - sampled_action.pow(2)) + 1e-6)
        log_probs = log_probs.sum(dim=1, keepdim=True)
    
        #could get it to return mean as the 'optimal' action during evaluation?
        return scaled_action, log_probs


class SACValueNetwork(nn.Module):
    def __init__(self, input_dim:int=256, hidden_dim=256):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        super(SACValueNetwork, self).__init__()
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
        self.apply(initialize_weights_xavier)

    def forward(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        # Assuming batch is dim=0, so state is shape [batch,state_space]
        # action is [batch,action_space]

        network_input = torch.cat([state,action],dim=1)
        action_value_estimate1 = self.ann1(network_input) # estimate the value of an action in a state
        action_value_estimate2 = self.ann2(network_input) # estimate the value of an action in a state

        return action_value_estimate1,action_value_estimate2

#OLD NOT WORKING
# class SACAgent(Agent):

#     def __init__(self, env: gym.Env, update_threshold: int = 1000, batch_size: int = 256, lr: float = 3e-4, gamma: float = 0.99, polyak = 0.995,alpha:float = 0.2):
#         # line 1 of pseudocode
#         self.loss_history = [0]
#         self.loss_history_actor = [0]
#         self.some_values = []
#         self.last_printed = 0.1

#         self.env = env
#         self.persistent_timesteps = 0

#         self.update_threshold = update_threshold
#         self.updates=0

#         # model hyperparams
#         self.batch_size = batch_size
#         self.alpha = alpha
#         self.lr = lr
#         self.gamma = gamma
#         self.polyak = polyak

#         observation_space_shape = env.observation_space._shape[0]
#         action_space_shape = env.action_space._shape[0]
#         action_space_max_value = env.action_space.high[0]

#         # self.replay_buffer = ReplayBuffer(1000000, observation_space_shape, action_space_shape)
        
#         self.actor = SACPolicyNetwork(input_dim=observation_space_shape, action_dim=action_space_shape,action_max=action_space_max_value)

#         self.critics = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)
#         self.critic_targets = SACValueNetwork(input_dim=observation_space_shape + action_space_shape)

#         self.polyak_update(0)


#         self.actor_loss = SACPolicyLoss()
#         self.critic_1_loss = nn.MSELoss()
#         self.critic_2_loss = nn.MSELoss()

#         self.actor_optimiser = optim.Adam(self.actor.parameters(),lr=self.lr)

#         self.critics_optimiser = optim.Adam(self.critics.parameters(),lr=self.lr)

#         # line 2 of pseudocode
#         # self.polyak_update(0)


#     def polyak_update(self, polyak):
#         for (parameter, target_parameter) in zip(self.critics.parameters(), self.critic_targets.parameters()):
#             target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)

#     def update_params(self,replay_buffer:ReplayBuffer):
#         # line 11 of pseudocode
#             old_states, new_states, actions, rewards, terminals = replay_buffer.get(self.batch_size)
#             old_states = torch.tensor(old_states, dtype=torch.float32)
#             new_states = torch.tensor(new_states, dtype=torch.float32)
#             actions = torch.tensor(actions, dtype=torch.float32)
#             rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
#             terminals = torch.tensor(terminals, dtype=torch.float32).unsqueeze(1)

#             # line 12 of pseudocode
#             with torch.no_grad():
#                 actor_prediction_new, log_actor_probability_new,_ = self.actor.sample(new_states)

#                 critic_target_1_prediction, critic_target_2_prediction = self.critic_targets.forward(new_states,actor_prediction_new)
#                 critic_target_clipped = torch.min(critic_target_1_prediction, critic_target_2_prediction)

#                 predicted_target_reward = critic_target_clipped - self.alpha * log_actor_probability_new
#                 target = rewards + self.gamma * (1 - terminals) * predicted_target_reward

#             # line 13 of pseudocode
#             critic_1_evaluation, critic_2_evaluation = self.critics.forward(old_states, actions)

#             critic_1_loss = self.critic_1_loss(critic_1_evaluation, target)
#             critic_2_loss = self.critic_2_loss(critic_2_evaluation, target)
#             total_critic_loss = critic_1_loss + critic_2_loss

#             self.critics_optimiser.zero_grad()

#             total_critic_loss.backward()
            
#             self.critics_optimiser.step()

#             # line 14 of pseudocode
#             actor_prediction_old, log_actor_probability_old,_ = self.actor.sample(old_states)
#             critic_1_prediction, critic_2_prediction = self.critics.forward(old_states, actor_prediction_old)
#             critic_clipped = torch.min(critic_1_prediction, critic_2_prediction)

#             actor_loss = self.actor_loss(critic_clipped, log_actor_probability_old, self.alpha)

#             self.actor_optimiser.zero_grad()
#             actor_loss.backward()
#             self.actor_optimiser.step()
#             ############
            
#             detached_loss = total_critic_loss.detach().numpy()

#             if detached_loss != self.loss_history[-1]:
#                 self.loss_history.append(detached_loss)

#             detached_actor_loss = actor_loss.detach().numpy()

#             if detached_actor_loss != self.loss_history_actor[-1]:
#                 self.loss_history_actor.append(detached_actor_loss)

#             self.polyak_update(self.polyak)

#     def simulate_episode(self,replay_buffer:ReplayBuffer,skip_update=False):
#         is_finished = False
#         is_truncated = False
#         reward_history = []

#         state, _ = self.env.reset()
#         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         timestep = 0

#         # line 3 of pseudocode
#         while True:
#             timestep += 1
#             self.persistent_timesteps += 1

#             # line 4 of pseudocode
#             if skip_update:
#                 action = self.env.action_space.sample()
#                 new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
#             else:
#                 action = self.actor.sample(state)[0].detach().numpy()
#                 new_state, reward, is_finished, is_truncated, _ = self.env.step(action[0])

#             # line 5-6 of pseudocode

#             # new_state, reward, is_finished, is_truncated, _ = self.env.step(action[0])

#             reward = reward # can scale

#             new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
#             reward_history.append(reward)

#             # line 7 of pseudocode
#             replay_buffer.add(Experience(state[0], new_state[0], action, reward, is_finished))

#             # line 8 of pseudocode
#             if is_finished or is_truncated:
#                 break

#             # line 9 of pseudocode
#             if skip_update or self.persistent_timesteps % self.update_threshold != 0:
#                 continue

#             # lines 11-15 of pseudocode
#             self.update_params(replay_buffer)
            
#             state = new_state
        
#         if self.loss_history[-1] != self.last_printed:
#             self.last_printed = self.loss_history[-1]
#             print(self.loss_history[-1])

#         return timestep, sum(reward_history), self.loss_history, self.loss_history_actor


#     def train(self,replay_buffer, num_timesteps=100000, print_interval=50, start_timesteps=1000):
#         """Train the agent over a given number of episodes."""
#         self.persistent_timesteps = 0
#         timesteps = 0
#         episodes = 0

#         while timesteps < start_timesteps:
#             elapsed_timesteps, _ , _, _ = self.simulate_episode(replay_buffer,skip_update=True)
#             timesteps += elapsed_timesteps
#             episodes += 1

#             if (episodes % print_interval == 0):
#                 print(f"Start timesteps {100 * timesteps / start_timesteps:.2f}% complete...")
        
#         super().train(replay_buffer,num_timesteps - timesteps, print_interval)



#     def predict(self, state):
#         """Predict the best action for the current state."""
#         raise NotImplementedError
    
#     def save(self, path):
#         """Save the agent's data to the path specified."""
#         raise NotImplementedError
    
#     def load(self, path):
#         """Load the data from the path specified."""
#         raise NotImplementedError


# env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
# replay_buffer = ReplayBuffer(1000000,env.observation_space.shape[0],env.action_space._shape[0])
# model = SACAgent(env)
# model.train(num_timesteps=200000,replay_buffer = replay_buffer)


#### SOMEONE ELSES TRAINING CODE
####

batch_size = 256
start_steps = 1000
updates_per_step = 1
num_steps = 1000001

env = gym.make("InvertedPendulum-v4")

# Agent
agent = SACAgent(env)
state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.shape[0]
memory = ReplayBuffer(1000000,state_space_size,action_space_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state,_ = env.reset()

    while not done and episode_steps<5000:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action,_ = agent.actor.sample(state)
            action = action[0].detach().numpy()
            # action = agent.select_action(state).flatten()

        if len(memory) > batch_size:
            # Number of updates per step in environment
            for i in range(updates_per_step):
                # Update parameters of all the networks
                agent.update_params(memory)

                updates += 1

        next_state, reward, done, _, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        to_add = Experience(state.flatten(),next_state,action,reward,mask)

        # memory.add(state.flatten(), action, reward, next_state, mask) # Append transition to memory
        memory.add(to_add) # Append transition to memory


        state = next_state

    if total_numsteps > num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

env.close()