import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import ReLU
import gymnasium as gym
import copy
from typing import Any, NamedTuple, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from agent import Agent

GLOBAL_TIMESTEPS = []
GLOBAL_REWARDS = []

random.seed(0)

def make_video_ddpg(env_name,agent,save_path):
    video_env = gym.make(env_name,render_mode="rgb_array")
    print(f"Making video at {save_path}")
    frames = []
    state, _ = video_env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        frame = video_env.render()
        frames.append(frame)

        # action = agent.predict(torch.Tensor(state))
        # state, reward, done, truncated, info = env.step(action)
        action = agent.actor.predict(torch.Tensor([state]),test=False)
        state, reward, done, truncated ,info = video_env.step(action[0].detach().numpy())

    # Save frames as a video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    video_env.close()

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

    hidden_size = 256

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
            num_train_episodes: int = 100000,
            make_video: bool = False
    ):
        super(DDPGAgent, self).__init__(env)
        # set hyperparams
        self.max_buffer_size = max_buffer_size
        self.replay_sample_size = replay_sample_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.gamma = gamma
        self.training_frequency = training_frequency
        self.num_train_episodes = num_train_episodes
        self.make_video = make_video

        # set up environment
        self.env = env
        action_dim: int = self.env.action_space.shape[0]
        act_limit_high: int = self.env.action_space.high[0]
        act_limit_low: int = self.env.action_space.low[0]
        state_dim: int = self.env.observation_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.max_buffer_size, state_dim, action_dim)

        # policy
        self.actor = ActorNetwork(
            self.hidden_size,
            ReLU(),
            action_dim,
            state_dim,
            act_limit_high,
            act_limit_low
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

        # target
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # freeze target actor and target critic
        for parameter in self.target_actor.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        self.critic_losses = []
        self.actor_losses = []

    def predict(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()
    
    def train(self, num_timesteps=50000, start_timesteps=1000):
        """Train the agent over a given number of episodes."""
        self.persistent_timesteps = 0
        timesteps = 0
        timestep_list = []
        reward_list = []

        print(f"Populating replay buffer with {start_timesteps} timesteps of experience...")

        while timesteps < start_timesteps:
            elapsed_timesteps, start_rewards = self.simulate_episode(should_learn=False)
            timesteps += elapsed_timesteps

            timestep_list.append(timesteps)
            reward_list.append(start_rewards)
                
        super().train(num_timesteps=num_timesteps, start_timesteps=start_timesteps)


    def simulate_episode(self, should_learn=True):
        state, _ = self.env.reset()
        reward_total = 0
        timestep = 0
        a_loss, c_loss = 0, 0

        # line 3 of pseudocode
        while True:

            timestep += 1
            self.persistent_timesteps += 1

            # line 4 of pseudocode
            if should_learn:
                with torch.no_grad():
                    action = self.actor.predict(state).numpy()
            else:
                action = self.env.action_space.sample()


            # line 5-6 of pseudocode
            new_state, reward, is_finished, is_truncated, _ = self.env.step(action)
            reward_total += reward

            # line 7 of pseudocode
            self.replay_buffer.add(Experience(state, new_state, action, reward, is_finished))
            state = new_state

            # line 9-10 of pseudocode
            if should_learn and timestep % self.training_frequency == 0:
                a_loss, c_loss = self.update_params()

            # line 8 of pseudocode - swapped since we might as well update even if it's a terminal state and the logic is easier this way
            if is_finished or is_truncated:
                break

        return timestep, reward_total


    def update_params(self):

        old_states, new_states, actions, rewards, terminals = self.replay_buffer.get(self.replay_sample_size)


        self.actor_optimiser.zero_grad()
        self.critic_optimiser.zero_grad()

        actor_loss = 0
        critic_loss = 0


        old_states = torch.tensor(old_states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)

        critic_pred = self.critic.predict(old_states,actions).T


        # need this or else temp tuning doesnt work, also speeds it up
        with torch.no_grad():
            # line 12 of pseudocode
            actor_prediction_new = self.actor.predict(new_states).T

            target = rewards + self.gamma * (1 - terminals) * actor_prediction_new

        # line 13 of psuedocode
        critic_loss = torch.mean((critic_pred - target)**2)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # line 14 of psuedocode
        actor_prediction_old = self.actor.predict(old_states)
        critic_evalution = self.critic.predict(old_states, actor_prediction_old)
        actor_loss = torch.mean(-critic_evalution)

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        self.polyak_update()

        return actor_loss.detach().numpy().item(), critic_loss.detach().numpy().item()


    def polyak_update(self):
        for (parameter, target_parameter) in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_parameter.data.copy_((1 - self.polyak) * parameter.data + self.polyak * target_parameter.data)
        for (parameter, target_parameter) in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_parameter.data.copy_((1 - self.polyak) * parameter.data + self.polyak * target_parameter.data)

class ActorNetwork(nn.Module):

    def __init__(self, hidden_size, activation, action_dim, state_dim, action_lim_high, action_lim_low):
        super().__init__()
        self.action_max = action_lim_high
        self.action_min = action_lim_low
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
        action_range = (self.action_max - self.action_min) / 2.0
        action_mid = (self.action_max + self.action_min) / 2.0
        scaled_output = action_mid + action_range * x
        return scaled_output
    

    def predict(self, state, test=False):
        noise_rate = 0.1
        a = self.forward(torch.as_tensor(state, dtype=torch.float32))

        if not test:
            noise = torch.randn(a.shape)
            a += noise * noise_rate

        assert isinstance(a, torch.Tensor), f"Expected a PyTorch tensor, but got {type(a)}"
        assert a.dtype == torch.float32, f"Expected tensor of dtype float32, but got {a.dtype}"
        return a

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

        return self.forward(x)


if __name__ == "__main__":
    env = gym.make("Ant-v4", render_mode=None)
    agent = DDPGAgent(env)
    agent.train(10000)
