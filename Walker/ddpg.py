import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import ReLU
import gymnasium as gym
import copy
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from agent import Agent

GLOBAL_TIMESTEPS = []
GLOBAL_REWARDS = []

random.seed(0)

# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/core.py
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/ddpg.py

class ReplayBuffer:

    def __init__(self, max_buffer_size, sample_size):
        self.max_buffer_size = max_buffer_size
        self.sample_size = sample_size
        self.buffer: List[Tuple] = []

    def add(self, observation):
        if len(self.buffer) >= self.max_buffer_size:
            # randomly select an index
            index = random.randint(0, len(self.buffer) - 1)
            self.buffer[index] = observation
        else:
            self.buffer.append(observation)

    def sample(self):
        sample = []
        indices_selected = set()
        for i in range(self.sample_size):
            index = random.randint(0, len(self.buffer) - 1)
            while index in indices_selected:
                index = random.randint(0, len(self.buffer) - 1)
            sample.append(self.buffer[index])
        return sample

class DDPGAgent(Agent):

    hidden_size = (256, 256)

    def __init__(
            self,
            env,
            max_buffer_size: int = 100000,
            replay_sample_size: int = 10,
            actor_lr: float = 0.0001,
            critic_lr: float = 0.0001,
            polyak: float = 0.995,
            gamma: float = 0.99,
            training_frequency: int = 10
    ):
        # set hyperparams
        self.max_buffer_size = max_buffer_size
        self.replay_sample_size = replay_sample_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.gamma = gamma
        self.training_frequency = training_frequency

        # set up environment
        self.env = env
        action_dim: int = self.env.action_space.shape[0]
        act_limit_high: int = self.env.action_space.high[0]
        act_limit_low: int = self.env.action_space.low[0]
        state_dim: int = self.env.observation_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.max_buffer_size, self.replay_sample_size)

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
            state_dim,
            action_dim
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

    def critic_loss(self, data):
        # extract observations from data
        i = 0
        loss = 0
        for observation in data:
            current_state, action, reward, next_state, terminal = observation
            pred = self.critic.get_q_value(current_state, action)
            loss += (pred - (reward+ self.gamma*(1 - terminal)*self.target_critic.get_q_value(next_state, self.target_actor.get_action(next_state))))**2
            i += 1
            
        if (i != 0):
            loss = loss / i

        return loss

    def actor_loss(self, data):
        i = 0
        loss = 0
        for observation in data: 
            current_state, action, reward, next_state, terminal = observation
            loss += -(self.critic.get_q_value(current_state, self.actor.get_action(current_state)))
            i += 1

        if (i != 0):
            loss = loss / i

        return loss 

    def train(self, num_train_episodes=1000000, start_steps=100):

        last_s, _ = self.env.reset()

        for episode in tqdm(range(start_steps)):
            rand_a = self.env.action_space.sample()
            new_s, reward, terminated, truncated, *args = self.env.step(rand_a)
            done = terminated or truncated
            self.replay_buffer.add((last_s, rand_a, reward, new_s, done))
            if done:
                last_s, _ = self.env.reset()
            else:
                last_s = new_s

        episodic_rewards = []

        total_reward = 0
        print("START TRAINING")
        alive = 0
        lives = 0
        for episode in tqdm(range(num_train_episodes)):

            # action -> numpy array
            a = self.actor.get_action(last_s).detach().numpy()

            # NUMPY ARRAY
            assert isinstance(a, np.ndarray), f"Expected a NumPy array, but got {type(a)}"

            new_s, reward, terminated, truncated, *args = self.env.step(a)
            total_reward += reward
            done = terminated or truncated
            self.replay_buffer.add((last_s, a, reward, new_s, done))
            if done:
                last_s, _ = self.env.reset()
                episodic_rewards.append(total_reward)
                GLOBAL_REWARDS.append(total_reward)
                GLOBAL_TIMESTEPS.append(episode)
                # print(lives, "attempt:\n", "died after ", alive, " steps", "total reward", total_reward, "\n")
                total_reward = 0
                alive = 0
                lives += 1
                
            else:
                alive += 1
                last_s = new_s

            if episode % self.training_frequency == 0:
                self.update_weights()

        episodic_rewards = []

        total_reward = 0
        print("START TESTING")
        alive = 0
        lives = 0
        self.critic_losses = []
        self.actor_losses = []

        for episode in tqdm(range(1000)):

            # action -> numpy array
            a = self.actor.get_action(last_s, test=True).detach().numpy()

            # NUMPY ARRAY
            assert isinstance(a, np.ndarray), f"Expected a NumPy array, but got {type(a)}"

            new_s, reward, terminated, truncated, *args = self.env.step(a)
            total_reward += reward
            done = terminated or truncated
            #self.replay_buffer.add((last_s, a, reward, new_s, done))
            if done:
                last_s, _ = self.env.reset()
                episodic_rewards.append(total_reward)
                total_reward = 0
                alive = 0
                lives += 1
                
            else:
                alive += 1
                last_s = new_s

        
        # # Plot the episodic curve
        # plt.plot(range(len(episodic_rewards)), episodic_rewards, label="Episodic rewards")
        # plt.xlabel("Episodes")
        # plt.ylabel("Total reward")
        # plt.legend()
        # plt.show()

        # # Plot the critic loss curve
        # plt.plot(range(len(self.critic_losses)), self.critic_losses, label="Critic loss")
        # plt.xlabel("Episodes")
        # plt.ylabel("Critic Losses")
        # plt.legend()
        # plt.show()

        # # Plot the actor loss curve
        # plt.plot(range(len(self.actor_losses)), self.actor_losses, label="Actor loss")
        # plt.xlabel("Episodes")
        # plt.ylabel("Actor Losses")
        # plt.legend()
        # plt.show()

    def update_weights(self):
        samples = self.replay_buffer.sample()
        self.actor_optimiser.zero_grad()
        self.critic_optimiser.zero_grad()

        # compute critic loss
        critic_loss = self.critic_loss(samples)
        self.critic_losses.append(critic_loss.item())
        critic_loss.backward()
        self.critic_optimiser.step()

        # freeze the crtic
        for parameter in self.critic.parameters():
            parameter.requires_grad = False

        # actor loss computation
        actor_loss = self.actor_loss(samples)
        self.actor_losses.append(actor_loss.item())
        actor_loss.backward()
        self.actor_optimiser.step()

        # unfreeze the critic
        for parameter in self.critic.parameters():
            parameter.requires_grad = True

        # polyak averaging -> actor params
        for actor_p, target_actor_p in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_p.data.mul_(self.polyak)
            target_actor_p.data.add_((1 - self.polyak) * actor_p.data)

        # polyak averaging -> critic params
        for critic_p, target_critic_p in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_p.data.mul_(self.polyak)
            target_critic_p.data.add_((1 - self.polyak) * critic_p.data)

class ActorNetwork(nn.Module):

    def __init__(self, hidden_size, activation, action_dim, state_dim, action_lim_high, action_lim_low):
        super().__init__()
        self.action_max = action_lim_high
        self.action_min = action_lim_low
        input_size = state_dim
        output_size = action_dim
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size[0])) # input layer
        layers.append(activation)

        for i in range(0, len(hidden_size) - 2):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size[-1], output_size)) # output layer

        self.network = nn.Sequential(*layers) # unpack layers and activation functions into sequential

    def forward(self, x):
        network_output = self.network(x)
        x = torch.tanh(network_output) # we need to scale the network's output within the action range
        action_range = (self.action_max - self.action_min) / 2.0
        action_mid = (self.action_max + self.action_min) / 2.0
        scaled_output = action_mid + action_range * x
        return scaled_output
    

    def get_action(self, state, test=False):
        noise_rate = 0.9
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
        output_size = 1 # critic network just outputs a value
        layers = []

        layers.append(nn.Linear(input_size, hidden_size[0])) # input layer
        layers.append(activation)

        for i in range(0, len(hidden_size) - 2):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size[-1], output_size)) # output layer

        self.network = nn.Sequential(*layers) # unpack layers and activation functions into sequential

    def forward(self, x):
        return self.network(x)
    
    def get_q_value(self, state, action):

        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        x = torch.concatenate((state, action))
        return self.forward(x)
    

def render_agent(env, agent, num_episodes=5):
    """
    Visualize the trained agent in the environment.
    Args:
        env: The gym environment.
        agent: The trained DDPG agent.
        num_episodes: Number of episodes to render.
    """
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        print(f"Episode {episode + 1}/{num_episodes}")
        while not done:
            env.render()  # Render the environment
            action = agent.actor.get_action(state, test=True).detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        print(f"Total reward for episode {episode + 1}: {total_reward}")
    env.close()


# if __name__ == "__main__":

#     env = gym.make("Ant-v4", render_mode=None)
#     agent = DDPGAgent(env)
#     agent.train(10000)
#     env = gym.make("Ant-v4", render_mode="rgb_array")
#     render_agent(env, agent, num_episodes=10)
