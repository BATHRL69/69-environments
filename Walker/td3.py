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

random.seed(0)

# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/core.py
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/ddpg.py

GLOBAL_TIMESTEPS = []
GLOBAL_REWARDS = []


def make_video_td3(env_name,agent,save_path):
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
        action = agent.actor.get_action(torch.Tensor([state]),test=False)
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

# class ReplayBuffer:

#     def __init__(self, max_buffer_size, sample_size):
#         self.max_buffer_size = max_buffer_size
#         self.sample_size = sample_size
#         self.buffer: List[Tuple] = []

#     def add(self, observation):
#         if len(self.buffer) >= self.max_buffer_size:
#             # randomly select an index
#             index = random.randint(0, len(self.buffer) - 1)
#             self.buffer[index] = observation
#         else:
#             self.buffer.append(observation)

#     def sample(self):
#         sample = []
#         indices_selected = set()
#         for i in range(self.sample_size):
#             index = random.randint(0, len(self.buffer) - 1)
#             while index in indices_selected:
#                 index = random.randint(0, len(self.buffer) - 1)
#             sample.append(self.buffer[index])
#         return sample

class TD3Agent(Agent):

    # hidden_size = (256, 256)
    hidden_size = (400, 300)

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
            num_train_episodes: int = 100000,
            make_video = False
    ):
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
        self.num_train_episodes = num_train_episodes
        self.make_video = make_video

        # set up environment
        self.env = env
        action_dim: int = self.env.action_space.shape[0]
        self.act_limit_high: int = self.env.action_space.high[0]
        self.act_limit_low: int = self.env.action_space.low[0]
        state_dim: int = self.env.observation_space.shape[0]

        self.replay_buffer = ReplayBuffer(self.max_buffer_size,state_dim,action_dim)


        # policy
        self.actor = ActorNetwork(
            self.hidden_size,
            ReLU(),
            action_dim,
            state_dim,
            self.act_limit_high,
            self.act_limit_low
        )
        self.actor_optimiser = Adam(self.actor.parameters(), lr = self.actor_lr)

        # q-value function 1
        self.critic_1 = CriticNetwork(
            self.hidden_size,
            ReLU(),
            state_dim,
            action_dim
        )
        self.critic_optimiser_1 = Adam(self.critic_1.parameters(), lr = self.critic_lr)

        # q-value function 2
        self.critic_2 = CriticNetwork(
            self.hidden_size,
            ReLU(),
            state_dim,
            action_dim
        )
        self.critic_optimiser_2 = Adam(self.critic_2.parameters(), lr = self.critic_lr)

        # target
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        # freeze target actor and target critic
        for parameter in self.target_actor.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic_1.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic_2.parameters():
            parameter.requires_grad = False

        self.critic_losses_1 = []
        self.critic_losses_2 = []
        self.actor_losses = []

    def predict(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()

    def get_critic_losses(self, data):
        # extract observations from data
        loss_q1 = 0
        loss_q2 = 0
        #this is batched now
        current_states, next_states, actions, rewards, terminals = data
        current_states = torch.Tensor(current_states)
        next_states = torch.Tensor(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards).unsqueeze(1)
        terminals = torch.Tensor(terminals).unsqueeze(1)

        pred_q1 = self.critic_1.get_q_value(current_states, actions)
        pred_q2 = self.critic_2.get_q_value(current_states, actions)

        eps = torch.randn(actions.shape) * self.actor_target_noise
        eps = torch.clamp(eps, -self.actor_noise_clip, self.actor_noise_clip)

        noised_target_action = self.target_actor.get_action(next_states) + eps
        noised_target_action = torch.clamp(noised_target_action, self.act_limit_low, self.act_limit_high)

        true = (rewards + self.gamma*(1 - terminals)*self.get_min_target_q_value(next_states, noised_target_action))

        loss_q1 += (pred_q1 - true)**2
        loss_q2 += (pred_q2 - true)**2
            
        loss_q1 = torch.mean(loss_q1)
        loss_q2 = torch.mean(loss_q2)

        return loss_q1, loss_q2
    

    def get_min_target_q_value(self, state, action):
        q_value_1 = self.target_critic_1.get_q_value(state, action)
        q_value_2 = self.target_critic_2.get_q_value(state, action)

        min_q_value = torch.min(q_value_1, q_value_2) # change if gradient issues?

        return min_q_value
    

    def get_min_q_value(self, state, action):
        q_value_1 = self.critic_1.get_q_value(state, action)
        q_value_2 = self.critic_2.get_q_value(state, action)

        min_q_value = torch.min(q_value_1, q_value_2) # change if gradient issues?

        return min_q_value
    

    def actor_loss(self, data):
        loss = 0
        #this is batched now
        current_state, next_state, action, reward, terminal = data
        loss += -(self.get_min_q_value(current_state, self.actor.get_action(current_state)))

        loss = torch.mean(loss)

        return loss 

    def train(self, start_steps=100):

        last_s, _ = self.env.reset()

        for episode in tqdm(range(start_steps)):
            rand_a = self.env.action_space.sample()
            new_s, reward, terminated, truncated, *args = self.env.step(rand_a)
            done = terminated or truncated
            experience = Experience(last_s,new_s,rand_a,reward,done)
            self.replay_buffer.add(experience)
            if done:
                last_s, _ = self.env.reset()
            else:
                last_s = new_s

        episodic_rewards = []

        total_reward = 0
        print("START TRAINING")
        alive = 0
        lives = 0
        for episode in tqdm(range(self.num_train_episodes)):
            
            if episode in {10000,100000,200000,500000,999999} and self.make_video:
                save_path_str = "new_td3_ant_"+str(episode)+".mp4"
                make_video_td3("Ant-v4",self,save_path_str)
            # action -> numpy array
            a = self.actor.get_action(last_s).detach().numpy()

            # NUMPY ARRAY
            assert isinstance(a, np.ndarray), f"Expected a NumPy array, but got {type(a)}"

            new_s, reward, terminated, truncated, *args = self.env.step(a)
            total_reward += reward
            done = terminated or truncated
            experience = Experience(last_s,new_s,a,reward,done)
            self.replay_buffer.add(experience)
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
                self.update_weights((episode % self.actor_update_frequency == 0))

        episodic_rewards = []

        total_reward = 0
        print("START TESTING")
        alive = 0
        lives = 0
        self.critic_losses_1 = []
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
        # plt.plot(range(len(self.critic_losses_1)), self.critic_losses_1, label="Critic loss")
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

    def update_weights(self, update_actor):
        
        samples = self.replay_buffer.get(self.replay_sample_size)
        self.actor_optimiser.zero_grad()
        self.critic_optimiser_1.zero_grad()
        self.critic_optimiser_2.zero_grad()


        # compute critic loss
        critic_loss_1, critic_loss_2 = self.get_critic_losses(samples)

        # update critic 1
        self.critic_losses_1.append(critic_loss_1.item())
        critic_loss_1.backward()
        self.critic_optimiser_1.step()

        # update critic 2
        self.critic_losses_2.append(critic_loss_2.item())
        critic_loss_2.backward()
        self.critic_optimiser_2.step()

        if (update_actor):
            # freeze both critics
            for parameter in self.critic_1.parameters():
                parameter.requires_grad = False

            for parameter in self.critic_2.parameters():
                parameter.requires_grad = False
            
            # actor loss computation
            actor_loss = self.actor_loss(samples)
            self.actor_losses.append(actor_loss.item())

            actor_loss.backward()
            self.actor_optimiser.step()

            # unfreeze both critics
            for parameter in self.critic_1.parameters():
                parameter.requires_grad = True

            for parameter in self.critic_2.parameters():
                parameter.requires_grad = True

            # polyak averaging -> actor params
            for actor_p, target_actor_p in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_actor_p.data.mul_(self.polyak)
                target_actor_p.data.add_((1 - self.polyak) * actor_p.data)

            # polyak averaging -> critic 1 params
            for critic_p, target_critic_p in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_critic_p.data.mul_(self.polyak)
                target_critic_p.data.add_((1 - self.polyak) * critic_p.data)

            for critic_p, target_critic_p in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
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

        for i in range(0, len(hidden_size) - 1):
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
        output_size = 1 # critic network just outputs a value
        layers = []

        layers.append(nn.Linear(input_size, hidden_size[0])) # input layer
        layers.append(activation)

        for i in range(0, len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size[-1], output_size)) # output layer

        self.network = nn.Sequential(*layers) # unpack layers and activation functions into sequential

    def forward(self, x):
        return self.network(x)
    
    def get_q_value(self, state, action):

        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        x = torch.concatenate([state, action],dim=1)
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


if __name__ == "__main__":

    env = gym.make("Ant-v4", render_mode=None)
    agent = TD3Agent(env)
    agent.train()
    env = gym.make("Ant-v4", render_mode="human")
    render_agent(env, agent, num_episodes=10)
