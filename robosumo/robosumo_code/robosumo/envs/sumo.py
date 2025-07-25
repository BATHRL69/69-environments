"""
Multi-agent sumo environment.
"""

import os
import tempfile

import numpy as np

import gym
from gym.spaces import Tuple
from gym.utils import EzPickle

import mujoco

from .mujoco_env import MujocoEnv  # Adjust the import to your project structure
from . import agents
from .utils import construct_scene

_AGENTS = {
    'ant': os.path.join(os.path.dirname(__file__), "assets", "ant.xml"),
    'bug': os.path.join(os.path.dirname(__file__), "assets", "bug.xml"),
    'spider': os.path.join(os.path.dirname(__file__), "assets", "spider.xml"),
}


class SumoEnv(MujocoEnv, EzPickle):
    """
    Multi-agent sumo environment.

    The goal of each agent is to push the other agent outside the tatami area.
    The reward is shaped such that agents learn to prefer staying in the center
    and pushing each other further away from the center. If any of the agents
    gets outside of the tatami (even accidentally), it gets -WIN_REWARD
    and the opponent gets +WIN_REWARD.
    """
    WIN_REWARD = 2000.0
    DRAW_PENALTY = -1000.0
    STAY_IN_CENTER_COEF = 0.1
    MOVE_TO_OPP_COEF = 0.1
    PUSH_OUT_COEF = 10.0

    def __init__(self, agent_names,
                 xml_path=None,
                 init_pos_noise=0.1,
                 init_vel_noise=0.1,
                 agent_kwargs=None,
                 frame_skip=5,
                 tatami_size=2.0,
                 timestep_limit=500,
                 **kwargs):
        EzPickle.__init__(self)
        self._tatami_size = tatami_size + 0.1
        self._timestep_limit = timestep_limit
        self._init_pos_noise = init_pos_noise
        self._init_vel_noise = init_vel_noise
        self._n_agents = len(agent_names)
        self._mujoco_init = False
        self._num_steps = 0
        self._spec = None

        # Resolve agent scopes
        agent_scopes = [
            f"{name}{i}"
            for i, name in enumerate(agent_names)
        ]

        # Construct scene XML
        scene_xml_path = os.path.join(os.path.dirname(__file__),
                                      "assets", "tatami.xml")
        agent_xml_paths = [_AGENTS[name] for name in agent_names]
        scene = construct_scene(scene_xml_path, agent_xml_paths,
                                agent_scopes=agent_scopes,
                                tatami_size=tatami_size,
                                **kwargs)

        # Initialize MuJoCo
        if xml_path is None:
            with tempfile.TemporaryDirectory() as tmpdir_name:
                scene_filepath = os.path.join(tmpdir_name, "scene.xml")
                scene.write(scene_filepath)
                super().__init__(model_path=scene_filepath, frame_skip=frame_skip)
        else:
            with open(xml_path, 'w') as fp:
                scene.write(fp.name)
            super().__init__(model_path=fp.name, frame_skip=frame_skip)
        self._mujoco_init = True

        # Construct agents
        agent_kwargs = agent_kwargs or {}
        self.agents = [
            agents.get_agent(name, model=self.model, data=self.data, scope=agent_scopes[i], **agent_kwargs)
            for i, name in enumerate(agent_names)
        ]

        # Set opponents
        for i, agent in enumerate(self.agents):
            agent.set_opponents([
                opp_agent for j, opp_agent in enumerate(self.agents) if j != i
            ])

        # Setup agents
        for agent in self.agents:
            agent.setup_spaces()

        # Set observation and action spaces
        self.observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])
        self.action_space = Tuple([
            agent.action_space for agent in self.agents
        ])

    def simulate(self, actions):
        a = np.concatenate(actions, axis=0)
        self.do_simulation(a, self.frame_skip)

    def step(self, actions):
        if not self._mujoco_init:
            return self._get_obs(), (0,) * self._n_agents, (False,) * self._n_agents, [{}] * self._n_agents

        dones = [False for _ in range(self._n_agents)]
        rewards = [0.0 for _ in range(self._n_agents)]
        infos = [{} for _ in range(self._n_agents)]

        # Call `before_step` on the agents
        for agent in self.agents:
            agent.before_step()

        # Do simulation
        self.simulate(actions)

        # Call `after_step` on the agents
        for i, agent in enumerate(self.agents):
            infos[i]['ctrl_reward'] = agent.after_step(actions[i])

        # Get observations
        obs = self._get_obs()

        self._num_steps += 1

        # Compute rewards and dones
        for i, agent in enumerate(self.agents):
            self_xyz = agent.get_qpos()[:3]
            # Lose penalty
            infos[i]['lose_penalty'] = 0.0
            if (self_xyz[2] < 0.29 or
                    np.max(np.abs(self_xyz[:2])) >= self._tatami_size):
                infos[i]['lose_penalty'] = -self.WIN_REWARD
                dones[i] = True
            # Win reward
            infos[i]['win_reward'] = 0.0
            for opp in agent._opponents:
                opp_xyz = opp.get_qpos()[:3]
                if (opp_xyz[2] < 0.29 or
                        np.max(np.abs(opp_xyz[:2])) >= self._tatami_size):
                    infos[i]['win_reward'] += self.WIN_REWARD
                    infos[i]['winner'] = True
                    dones[i] = True
            infos[i]['main_reward'] = \
                infos[i]['win_reward'] + infos[i]['lose_penalty']
            # Draw penalty
            if self._num_steps > self._timestep_limit:
                infos[i]['main_reward'] += self.DRAW_PENALTY
                dones[i] = True
            # Move to opponent(s) and push them out of center
            infos[i]['move_to_opp_reward'] = 0.0
            infos[i]['push_opp_reward'] = 0.0
            for opp in agent._opponents:
                infos[i]['move_to_opp_reward'] += \
                    self._comp_move_reward(agent, opp.posafter)
                infos[i]['push_opp_reward'] += \
                    self._comp_push_reward(agent, opp.posafter)
            # Reward shaping
            infos[i]['shaping_reward'] = \
                infos[i]['ctrl_reward'] + \
                infos[i]['push_opp_reward'] + \
                infos[i]['move_to_opp_reward']
            # Add up rewards
            rewards[i] = infos[i]['main_reward'] + infos[i]['shaping_reward']

        rewards = tuple(rewards)
        dones = tuple(dones)
        infos = tuple(infos)

        return obs, rewards, dones, infos

    def _comp_move_reward(self, agent, target):
        move_vec = (agent.posafter - agent.posbefore) / self.dt
        direction = target - agent.posbefore
        norm = np.linalg.norm(direction)
        if norm == 0:
            return 0.0
        direction /= norm
        return max(np.dot(move_vec, direction), 0.0) * self.MOVE_TO_OPP_COEF

    def _comp_push_reward(self, agent, target):
        dist_to_center = np.linalg.norm(target)
        return -self.PUSH_OUT_COEF * np.exp(-dist_to_center)

    def _get_obs(self):
        if not self._mujoco_init:
            return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return tuple(agent.get_obs() for agent in self.agents)

    def reset_model(self):
        self._num_steps = 0
        # Randomize agent positions
        r, z = 1.15, 1.25
        delta = (2.0 * np.pi) / self._n_agents
        phi = self.np_random.uniform(0.0, 2.0 * np.pi)
        for i, agent in enumerate(self.agents):
            angle = phi + i * delta
            x, y = r * np.cos(angle), r * np.sin(angle)
            agent.set_xyz((x, y, z))
        # Add noise to all qpos and qvel elements
        pos_noise = self.np_random.uniform(
            low=-self._init_pos_noise,
            high=self._init_pos_noise,
            size=self.model.nq)
        vel_noise = self._init_vel_noise * self.np_random.randn(self.model.nv)
        qpos = self.data.qpos.ravel().copy() + pos_noise
        qvel = self.data.qvel.ravel().copy() + vel_noise
        self.init_qpos, self.init_qvel = qpos, qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -25
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.distance = self.model.stat.extent * 1.0

    def render(self, mode='human'):
        return super().render(mode=mode)
