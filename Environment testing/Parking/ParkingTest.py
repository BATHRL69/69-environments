### Setup ###
# rllib installed from  : git clone git@github.com:SCP-CN-001/rllib.git
# Create venv, activate,  pip install -r requirements.txt
### Run ###

import sys

sys.path.append(".")
sys.path.append("./rllib")
sys.path.append("../..")
sys.path.append("../../tactics2d")

import os
import time
from collections import deque
import heapq
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from shapely.geometry import LinearRing, LineString, Point
import torch
from torch.distributions import Normal
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter

from tactics2d.envs import ParkingEnv
from tactics2d.math.interpolate import ReedsShepp
from tactics2d.traffic.status import ScenarioStatus


from rllib.algorithms.ppo import *  ## Environment

# the proportion of the type of parking lot,
# 0 means all scenarios are parallel parking, 1 means all scenarios are vertical parking
type_proportion = 1.0
# the render mode, "rgb_array" means render the scene to a numpy array, "human" means render the scene to a window
render_mode = ["rgb_array", "human"][0]
render_fps = 1000
# the max step of one episode
max_step = 1000
env = ParkingEnv(
    type_proportion=type_proportion,
    render_mode=render_mode,
    render_fps=render_fps,
    max_step=max_step,
)

num_lidar_rays = env.scenario_manager._lidar_line  # 360
lidr_obs_shape = num_lidar_rays // 3  # 120
lidar_range = env.scenario_manager._lidar_range


class ParkingWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_shape = (
            lidr_obs_shape + 6
        )  # 120: lidar obs size. 6: size of additional features we add
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_shape, 1), dtype=np.float32
        )

    def _preprocess_action(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1, 1)
        action_space = self.env.action_space
        action = (
            action * (action_space.high - action_space.low) / 2
            + (action_space.high + action_space.low) / 2
        )
        return action

    def _preprocess_observation(self, info):
        lidar_info = np.clip(info["lidar"], 0, 20)
        lidar_info = lidar_info[
            ::3
        ]  # we downsample the lidar data from 360 to 120 to feed into the model
        lidar_feature = lidar_info / 20.0  # normalize the lidar data to [0, 1]
        other_feature = np.array(
            [
                info["diff_position"]
                / 10.0,  # normalize the distance to target position
                np.cos(info["diff_angle"]),
                np.sin(info["diff_angle"]),
                np.cos(info["diff_heading"]),
                np.sin(info["diff_heading"]),
                info["state"].speed,
            ]
        )

        observation = np.concatenate([lidar_feature, other_feature])
        return observation

    def reset(self, seed: int = None, options: dict = None):
        _, info = self.env.reset(seed, options)
        custom_observation = self._preprocess_observation(info)
        return custom_observation, info

    def step(self, action):
        action = self._preprocess_action(action)
        _, reward, terminated, truncated, info = self.env.step(action)
        custom_observation = self._preprocess_observation(info)

        return custom_observation, reward, terminated, truncated, info


## Environment
# the proportion of the type of parking lot,
# 0 means all scenarios are parallel parking, 1 means all scenarios are vertical parking


# the wrapper is used to preprocess the observation and action


STEER_RATIO = 0.98
vehicle = env.scenario_manager.agent

env = ParkingWrapper(env)


max_speed = 0.5  # we manually set the max speed in the parking task


class PIDController:
    def __init__(self, target, Kp=0.1, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, target=None):
        if target is not None:
            self.target = target
        error = self.target - current_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self, target=None):
        if target is not None:
            self.target = target
        self.prev_error = 0
        self.integral = 0


def rear_center_coord(center_x, center_y, heading, lr):
    """calculate the rear wheel center position based on the center position and heading angle"""
    x = center_x - lr * np.cos(heading)
    y = center_y - lr * np.sin(heading)
    return x, y, heading


class ParkingActor(PPOActor):
    def get_dist(self, state):
        policy_dist = self.forward(state)
        mean = torch.clamp(policy_dist, -1, 1)
        std = self.log_std.expand_as(mean).exp()
        dist = Normal(mean, std)

        return dist

    def action(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dist = self.get_dist(state)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()
        return action, log_prob


class ParkingAgent(PPO):
    def __init__(self, config, device, max_speed=0.5, max_acceleration=2.0):
        super(ParkingAgent, self).__init__(config, device)
        self.accel_controller = PIDController(0, 2.0, 0.0, 0.0)
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.action_cnt = 0
        self.last_action = None

    def control_rlagent_action(self, info, action):
        """
        The network is trained to output the accel and steer ratio, we need to limit the speed to interact with the environment.
        """
        action_shape = action.shape
        if len(action_shape) > 1:
            assert action_shape[0] == 1
            action = action.squeeze(0)
        curr_v = info["state"].speed
        max_positive_v = self.max_speed
        max_negative_v = -self.max_speed
        max_accel = self.accel_controller.update(curr_v, max_positive_v)
        max_decel = self.accel_controller.update(curr_v, max_negative_v)
        target_a = np.clip(action[1] * self.max_acceleration, max_decel, max_accel)
        action[1] = target_a / self.max_acceleration

        return action.reshape(*action_shape)

    def get_action(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        action, log_prob = self.actor_net.action(states)
        value = self.critic_net(states)
        value = value.detach().cpu().numpy().flatten()

        return action, log_prob, value

    def choose_action(self, info, state):
        """
        Choose to execute the action from rl agent's output if rs path is not available, otherwise execute the action from rs agent.
        """

        action, log_prob, value = self.get_action(state)

        return action, log_prob, value

    def evaluate_action(self, states, actions: np.ndarray):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        log_prob, _ = self.actor_net.evaluate(states, actions)
        value = self.critic_net(states)

        log_prob = log_prob.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()

        return log_prob, value

    def reset(self):
        self.accel_controller.reset()


def train_rl_agent(env, agent, episode_num=int(1e5), log_path=None, verbose=True):
    if log_path is None:
        log_dir = "./logs"
        current_time = time.localtime()
        timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
        log_path = os.path.join(log_dir, timestamp)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    reward_list = deque(maxlen=100)
    success_list = deque(maxlen=100)
    loss_list = deque(maxlen=100)
    status_info = deque(maxlen=100)

    step_cnt = 0
    episode_cnt = 0

    print("start train!")
    while episode_cnt < episode_num:

        state, info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        episode_step_cnt = 0
        print(episode_cnt)

        while not done:
            step_cnt += 1
            episode_step_cnt += 1
            action, log_prob, value = agent.choose_action(info, state)
            if len(action.shape) == 2:
                action = action.squeeze(0)
            if len(log_prob.shape) == 2:
                log_prob = log_prob.squeeze(0)
            next_state, reward, terminate, truncated, info = env.step(action)
            # if episode_cnt % 10 == 0:
            env.render()
            done = terminate or truncated
            total_reward += reward
            observations = [[next_state], [reward], [terminate], [truncated], [info]]
            agent.push([observations, [state], [action], [log_prob], [value]])
            # early stop the episode if the vehicle could not find an available RS path
            if episode_step_cnt >= 400:
                done = True
                break

            state = next_state
            loss = agent.train()
            if loss is not None:
                loss_list.append(loss)

        status_info.append([info["scenario_status"], info["traffic_status"]])
        success_list.append(int(info["scenario_status"] == ScenarioStatus.COMPLETED))
        reward_list.append(total_reward)
        episode_cnt += 1

        if episode_cnt % 10 == 0:
            if verbose:
                print(
                    "episode: %d, total step: %d, average reward: %s, success rate: %s"
                    % (
                        episode_cnt,
                        step_cnt,
                        np.mean(reward_list),
                        np.mean(success_list),
                    )
                )
                print("last 10 episode:")
                for i in range(10):
                    print(reward_list[-(10 - i)], status_info[-(10 - i)])
                print("")

            writer.add_scalar("average_reward", np.mean(reward_list), episode_cnt)
            writer.add_scalar("average_loss", np.mean(loss_list), episode_cnt)
            writer.add_scalar("success_rate", np.mean(success_list), episode_cnt)

        if episode_cnt % 1000 == 0:
            agent.save(os.path.join(log_path, "model_%d.pth" % episode_cnt))


agent_config = PPOConfig(
    {
        "debug": False,
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "gamma": 0.995,
        "lr": 5e-6,
        "actor_net": ParkingActor,
        "actor_kwargs": {
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "hidden_size": 128,
            "continuous": True,
        },
        "critic_kwargs": {
            "state_dim": env.observation_space.shape[0],
            "hidden_size": 128,
        },
        "horizon": 20000,
        "batch_size": 128,
        "adam_epsilon": 1e-8,
    }
)
min_radius = vehicle.wheel_base / np.tan(vehicle.steer_range[1] * STEER_RATIO)
vehicle_rear_to_center = 0.5 * vehicle.length - vehicle.rear_overhang
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = ParkingAgent(agent_config, device)
log_path = "./logs"
num_episode = 100000

train_rl_agent(env, agent, episode_num=num_episode, log_path=log_path, verbose=True)


def eval_rl_agent(env, agent, episode_num=int(1e2), verbose=True):

    reward_list = deque(maxlen=episode_num)
    success_list = deque(maxlen=episode_num)
    loss_list = deque(maxlen=episode_num)
    status_info = deque(maxlen=episode_num)

    step_cnt = 0
    episode_cnt = 0

    print("start evaluation!")
    with torch.no_grad():
        while episode_cnt < episode_num:
            state, info = env.reset()
            agent.reset()
            done = False
            total_reward = 0
            episode_step_cnt = 0

            while not done:
                step_cnt += 1
                episode_step_cnt += 1
                action, log_prob, _ = agent.choose_action(info, state)
                if len(action.shape) == 2:
                    action = action.squeeze(0)
                if len(log_prob.shape) == 2:
                    log_prob = log_prob.squeeze(0)
                next_state, reward, terminate, truncated, info = env.step(action)
                env.render()
                done = terminate or truncated
                total_reward += reward
                state = next_state
                loss = agent.train()
                if not loss is None:
                    loss_list.append(loss)

            status_info.append([info["scenario_status"], info["traffic_status"]])
            success_list.append(
                int(info["scenario_status"] == ScenarioStatus.COMPLETED)
            )
            reward_list.append(total_reward)
            episode_cnt += 1

            if episode_cnt % 10 == 0:
                if verbose:
                    print(
                        "episode: %d, total step: %d, average reward: %s, success rate: %s"
                        % (
                            episode_cnt,
                            step_cnt,
                            np.mean(reward_list),
                            np.mean(success_list),
                        )
                    )
                    print("last 10 episode:")
                    for i in range(10):
                        print(reward_list[-(10 - i)], status_info[-(10 - i)])
                    print("")

    return np.mean(success_list), np.mean(reward_list)


start_t = time.time()
# agent.load("./data/parking_agent.pth")
succ_rate, avg_reard = eval_rl_agent(env, agent, episode_num=100, verbose=True)
print("Success rate: ", succ_rate)
print("Average reward: ", avg_reard)
print("eval time: ", time.time() - start_t)
