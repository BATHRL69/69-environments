import os
import tempfile
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Tuple
from gymnasium.utils import EzPickle

# Import the new MuJoCo package
import mujoco
from robosumo.envs import MujocoEnv

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
    """

    WIN_REWARD = 2000.
    DRAW_PENALTY = -1000.
    STAY_IN_CENTER_COEF = 0.1
    MOVE_TO_OPP_COEF = 0.1
    PUSH_OUT_COEF = 10.0

    def __init__(self, agent_names, xml_path=None, init_pos_noise=.1,
                 init_vel_noise=.1, agent_kwargs=None, frame_skip=5,
                 tatami_size=2.0, timestep_limit=500, **kwargs):
        EzPickle.__init__(self)
        self._tatami_size = tatami_size + 0.1
        self._timestep_limit = timestep_limit
        self._init_pos_noise = init_pos_noise
        self._init_vel_noise = init_vel_noise
        self._n_agents = len(agent_names)
        self._mujoco_init = False
        self._num_steps = None

        # Resolve agent scopes
        agent_scopes = [
            f"{name}{i}" for i, name in enumerate(agent_names)
        ]

        # Construct scene XML
        scene_xml_path = os.path.join(os.path.dirname(__file__), "assets", "tatami.xml")
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
                super().__init__(scene_filepath, frame_skip)
        else:
            with open(xml_path, 'w') as fp:
                scene.write(fp.name)
            super().__init__(fp.name, frame_skip)
        
        self._mujoco_init = True

        # Construct agents
        agent_kwargs = agent_kwargs or {}
        self.agents = [
            agents.get(name, env=self, scope=agent_scopes[i], **agent_kwargs)
            for i, name in enumerate(agent_names)
        ]

        # Set opponents
        for i, agent in enumerate(self.agents):
            agent.set_opponents([a for j, a in enumerate(self.agents) if j != i])

        # Setup agents
        for agent in self.agents:
            agent.setup_spaces()

        # Set observation and action spaces
        self.observation_space = Tuple([agent.observation_space for agent in self.agents])
        self.action_space = Tuple([agent.action_space for agent in self.agents])

    def simulate(self, actions):
        a = np.concatenate(actions, axis=0)
        self.do_simulation(a, self.frame_skip)

    def step(self, actions):
        # if not self._mujoco_init:
        #     return self._get_obs(), 0, False, {}, None

        dones = [False for _ in range(self._n_agents)]
        rewards = [0. for _ in range(self._n_agents)]
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
            # Loose penalty
            infos[i]['lose_penalty'] = 0.
            if (self_xyz[2] < 0.29 or
                    np.max(np.abs(self_xyz[:2])) >= self._tatami_size):
                infos[i]['lose_penalty'] = -self.WIN_REWARD
                dones[i] = True

            # Win reward
            infos[i]['win_reward'] = 0.
            for opp in agent._opponents:
                opp_xyz = opp.get_qpos()[:3]
                if (opp_xyz[2] < 0.29 or
                        np.max(np.abs(opp_xyz[:2])) >= self._tatami_size):
                    infos[i]['win_reward'] += self.WIN_REWARD
                    infos[i]['winner'] = True
                    dones[i] = True

            infos[i]['main_reward'] = infos[i]['win_reward'] + infos[i]['lose_penalty']

            # Draw penalty
            if self._num_steps > self._timestep_limit:
                infos[i]['main_reward'] += self.DRAW_PENALTY
                dones[i] = True

            # Move to opponent(s) and push them out of center
            infos[i]['move_to_opp_reward'] = sum(
                self._comp_move_reward(agent, opp.posafter) for opp in agent._opponents
            )
            infos[i]['push_opp_reward'] = sum(
                self._comp_push_reward(agent, opp.posafter) for opp in agent._opponents
            )

            # Reward shaping
            infos[i]['shaping_reward'] = infos[i]['ctrl_reward'] + infos[i]['push_opp_reward'] + infos[i]['move_to_opp_reward']

            # Add up rewards
            rewards[i] = infos[i]['main_reward'] + infos[i]['shaping_reward']

        rewards = tuple(rewards)
        dones = tuple(dones)
        infos = tuple(infos)

        # Gymnasium uses a single boolean for `done`, not a list
        return obs, rewards, any(dones), infos

    def reset(self, seed=None, options=None):
        self._num_steps = 0
        super().reset(seed=seed)

        # Randomize agent positions
        r, z = 1.15, 1.25
        delta = (2. * np.pi) / self._n_agents
        phi = self.np_random.uniform(0., 2. * np.pi)

        for i, agent in enumerate(self.agents):
            angle = phi + i * delta
            x, y = r * np.cos(angle), r * np.sin(angle)
            agent.set_xyz((x, y, z))

        # Add noise to all qpos and qvel elements
        pos_noise = self.np_random.uniform(size=self.model.nq, low=-self._init_pos_noise, high=self._init_pos_noise)
        vel_noise = self._init_vel_noise * self.np_random.randn(self.model.nv)

        qpos = self.data.qpos.ravel() + pos_noise
        qvel = self.data.qvel.ravel() + vel_noise

        self.set_state(qpos, qvel)
        return self._get_obs(), {}

    def viewer_setup(self):
        # Setting up the viewer using mujoco viewer settings
        if self.viewer:
            self.viewer.cam.trackbodyid = 0
            self.viewer.cam.elevation = -25
            self.viewer.cam.type = 1
            self.viewer.cam.distance = self.model.stat.extent * 1.0
