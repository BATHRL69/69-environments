import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import ReLU
import gymnasium as gym
from typing import Any, NamedTuple
import numpy as np

from agent import Agent

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

class DDPGAgent(Agent):


    def __init__(
            self,
            env,
            max_buffer_size: int = 1000000,
            replay_sample_size: int = 256,
            actor_lr: float = 0.0003,
            critic_lr: float = 0.0003,
            polyak: float = 0.995,
            gamma: float = 0.99,
            training_frequency: int = 1,
            make_video: bool = False
    ):
        super(DDPGAgent, self).__init__(env)
        # set hyperparams
        self.hidden_size=256
        self.max_buffer_size = max_buffer_size
        self.replay_sample_size = replay_sample_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.gamma = gamma
        self.training_frequency = training_frequency
        self.make_video = make_video

        # set up environment
        self.env = env
        self.action_limit: int = self.env.action_space.high[0]
        action_dim: int = self.env.action_space.shape[0]
        state_dim: int = self.env.observation_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.max_buffer_size, state_dim, action_dim)

        # policy
        self.actor = ActorNetwork(
            self.hidden_size,
            ReLU(),
            action_dim,
            state_dim,
            self.action_limit
        )

        self.actor_optimiser = Adam(self.actor.parameters(), lr = self.actor_lr)

        # q-value function
        self.critic = CriticNetwork(
            self.hidden_size,
            ReLU(),
            action_dim,
            state_dim
        )
        self.critic_optimiser = Adam(self.critic.parameters(), lr = self.critic_lr)

        # targets
        self.target_actor = ActorNetwork(
            self.hidden_size,
            ReLU(),
            action_dim,
            state_dim,
            self.action_limit
        )

        self.target_critic = CriticNetwork(
            self.hidden_size,
            ReLU(),
            action_dim,
            state_dim
        )

        # setting target parameters to be regular network parameters
        self.polyak_update(0)

    def predict(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()
    
    def train(self, num_timesteps=50000, start_timesteps=1000):
        """Train the agent over a given number of episodes."""
        self.persistent_timesteps = 0
        timesteps = 0

        print(f"Populating replay buffer with {start_timesteps} timesteps of experience...")

        # randomly sample from action space at first 
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

        while True:

            timestep += 1

            # either sample from action space, or get action from policy network (actor)
            if should_learn:
                with torch.no_grad():
                    action = self.actor.predict(state).numpy()
            else:
                action = self.env.action_space.sample()


            # sample experience from the environment (making the action)
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            reward_total += reward

            # save experience in replay buffer
            self.replay_buffer.add(Experience(state, new_state, action, reward, is_finished))
            state = new_state

            # update parameters if appropriate
            if should_learn and timestep % self.training_frequency == 0:
                self.update_params()

            # finish episode if its done 
            if is_finished or is_truncated:
                break

        return timestep, reward_total


    def update_params(self):

        # sampling from replay buffer
        old_states, new_states, actions, rewards, terminals = self.replay_buffer.get(self.replay_sample_size)
        old_states = torch.tensor(old_states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)

        critic_pred = self.critic.predict(old_states, actions)

        # stops target networks from being updated via gradient descent
        with torch.no_grad():
            
            # calculating the "true" value of the value of the state
            actor_prediction_new = self.target_actor.predict(new_states)
            target = rewards + self.gamma * (1 - terminals) * self.target_critic.predict(new_states, actor_prediction_new)


        # calculating critic loss + backpropping to update it
        critic_loss = (critic_pred - target)**2
        critic_loss = torch.mean(critic_loss)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # calculating actor loss + backpropping to update it
        actor_prediction_old = self.actor.predict(old_states)
        critic_evalution = self.critic.predict(old_states, actor_prediction_old)
        actor_loss = torch.mean(-critic_evalution)

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # polyak update target networks
        self.polyak_update(self.polyak)


    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)
        for (parameter, target_parameter) in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)

class ActorNetwork(nn.Module):

    def __init__(self, hidden_size, activation, action_dim, state_dim, action_limit):
        super().__init__()
        self.action_limit = action_limit
        input_size = state_dim
        output_size = action_dim

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        ) 

    def forward(self, x):
        network_output = self.network(x)
        x = torch.tanh(network_output) # we need to scale the network's output within the action range
        scaled_output = self.action_limit * x
        return scaled_output
    

    def predict(self, state, test=False):
        noise_rate = 0.1
        action = self.forward(torch.as_tensor(state, dtype=torch.float32))

        if not test:
            noise = torch.randn(action.shape)
            action += noise * noise_rate

        return action.squeeze() 

class CriticNetwork(nn.Module):
        
    def __init__(self, hidden_size, activation, action_dim, state_dim):
        super().__init__()
        input_size = action_dim + state_dim
        output_size = 1

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        ) 

    def forward(self, x):
        return self.network(x)
    
    def predict(self, state, action):

        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        x = torch.concatenate([state, action],dim=1)

        return self.forward(x).squeeze() 


if __name__ == "__main__":
    env = gym.make("Ant-v4", render_mode=None)
    agent = DDPGAgent(env)
    agent.train(10000)
