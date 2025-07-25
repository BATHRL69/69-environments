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

class TD3Agent(Agent):

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
            actor_update_frequency: int = 2,
            target_noise=0.2, 
            noise_clip=0.5,
    ):
        super(TD3Agent, self).__init__(env)
        # set hyperparams
        self.max_buffer_size = max_buffer_size
        self.replay_sample_size = replay_sample_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.gamma = gamma
        self.training_frequency = training_frequency
        self.actor_update_frequency = actor_update_frequency
        self.actor_target_noise = target_noise
        self.actor_noise_clip = noise_clip
        self.persistent_timesteps = 0

        # set up environment
        self.env = env
        action_dim: int = self.env.action_space.shape[0]
        self.action_limit: int = self.env.action_space.high[0]
        state_dim: int = self.env.observation_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.max_buffer_size,state_dim,action_dim)

        # policy
        self.actor = ActorNetwork(
            256,
            ReLU(),
            action_dim,
            state_dim,
            self.action_limit
        )
        self.actor_optimiser = Adam(self.actor.parameters(), lr = self.actor_lr)

        # q-value function 1
        self.critic_1 = CriticNetwork(
            256,
            ReLU(),
            state_dim,
            action_dim
        )
        # q-value function 2
        self.critic_2 = CriticNetwork(
            256,
            ReLU(),
            state_dim,
            action_dim
        )

        # only need one optimiser as we can just update both together
        self.critics_optimiser = Adam(list(self.critic_1.parameters())+list(self.critic_2.parameters()), lr=actor_lr)

        # targets
        self.target_actor = ActorNetwork(
            256,
            ReLU(),
            action_dim,
            state_dim,
            self.action_limit
        )

        self.target_critic_1 = CriticNetwork(
            256,
            ReLU(),
            action_dim,
            state_dim
        )

        self.target_critic_2 = CriticNetwork(
            256,
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
            self.persistent_timesteps += 1

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
            if should_learn and self.persistent_timesteps % self.training_frequency == 0:
                self.update_params(update_actor=timestep%self.actor_update_frequency)

            # finish episode if its done 
            if is_finished or is_truncated:
                break

        return timestep, reward_total
    
    def update_params(self, update_actor=True):

        # sampling from replay buffer
        old_states, new_states, actions, rewards, terminals = self.replay_buffer.get(self.replay_sample_size)
        old_states = torch.tensor(old_states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)

        # td3 improvement - have two critics
        critic_1_pred = self.critic_1.predict(old_states, actions)
        critic_2_pred = self.critic_2.predict(old_states, actions)

        # stops target networks from being updated via gradient descent
        with torch.no_grad():
            
            # calculating the "true" value of the value of the state
            actor_prediction_new = self.target_actor.predict(new_states)

            # td3 improvement - add noise to the action
            clipped_noise =  torch.clip(torch.randn(size=actor_prediction_new.shape)* self.actor_target_noise, -self.actor_noise_clip, self.actor_noise_clip)
            noised_actor_prediction = torch.clip(actor_prediction_new + clipped_noise, -self.action_limit, self.action_limit)

            # td3 improvement - take the minimum of both target critics
            critic_1_eval = self.target_critic_1.predict(new_states, noised_actor_prediction)
            critic_2_eval = self.target_critic_2.predict(new_states, noised_actor_prediction)
            target_critic_eval = torch.min(critic_1_eval, critic_2_eval)

            target = rewards + self.gamma * (1 - terminals) * target_critic_eval

        # calculating each critic loss + backpropping to update both
        critic_1_loss = torch.mean((critic_1_pred - target)**2)
        critic_2_loss = torch.mean((critic_2_pred - target)**2)
        total_critic_loss = critic_1_loss + critic_2_loss
        
        self.critics_optimiser.zero_grad()
        total_critic_loss.backward()
        self.critics_optimiser.step()

        # td3 improvement - delay policy network (actor) to only update every n critic updates
        if update_actor:

            # calculating actor loss + backpropping to update it
            actor_prediction_old = self.actor.predict(old_states)
            critic_evaluation = torch.min(self.critic_1.predict(old_states, actor_prediction_old), self.critic_2.predict(old_states, actor_prediction_old))
            actor_loss = torch.mean(-critic_evaluation)

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_optimiser.step()

            # we only update target networks when we also update policy
            self.polyak_update(self.polyak)

    def polyak_update(self, polyak):
        for (parameter, target_parameter) in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_parameter.data.copy_((1 - polyak) * parameter.data + polyak * target_parameter.data)

        for (parameter, target_parameter) in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
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
            nn.Linear( hidden_size, output_size)
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
        action = torch.clip(action,-self.action_limit,self.action_limit)
        return action.squeeze() 

class CriticNetwork(nn.Module):
        
    def __init__(self, hidden_size, activation, action_dim, state_dim):
        super().__init__()
        input_size = action_dim + state_dim
        output_size = 1

        self.network = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            activation,
            nn.Linear(hidden_size,hidden_size),
            activation,
            nn.Linear(hidden_size,output_size)
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
    agent = TD3Agent(env)
    agent.train(num_timesteps=200_000,start_timesteps=25000)
