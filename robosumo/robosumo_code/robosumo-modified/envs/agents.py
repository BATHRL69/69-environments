import os
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import gym


class Agent(object):
    """
    Superclass for all agents in sumo MuJoCo environment.
    """

    CFRC_CLIP = 100.

    COST_COEFS = {
        'ctrl': 1e-1,
        # 'pain': 1e-4,
        # 'attack': 1e-1,
    }

    JNT_NPOS = {
        0: 7,
        1: 4,
        2: 1,
        3: 1,
    }

    def __init__(self, env, scope, xml_path, adjust_z=0.):
        self._env = env
        self._scope = scope
        self._xml_path = xml_path
        self._xml = ET.parse(xml_path)
        self._adjust_z = adjust_z

        self._set_body()
        self._set_joint()

    def setup_spaces(self):
        self._set_observation_space()
        self._set_action_space()

    def _in_scope(self, name):
        return name.startswith(self._scope)

    def _set_body(self):
        self.body_names = list(filter(
            lambda x: self._in_scope(x), self._env.model.names[self._env.model.body_nameadr:]
        ))
        self.body_ids = [
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.body_names
        ]
        self.body_name_idx = {
            name.split('/')[-1]: idx
            for name, idx in zip(self.body_names, self.body_ids)
        }
        # Determine body params
        self.body_dofnum = self._env.model.body_dofnum[self.body_ids]
        self.body_dofadr = self._env.model.body_dofadr[self.body_ids]
        self.nv = self.body_dofnum.sum()
        # Determine qvel_start_idx and qvel_end_idx
        dof = list(filter(lambda x: x >= 0, self.body_dofadr))
        self.qvel_start_idx = int(dof[0])
        last_dof_body_id = self.body_dofnum.shape[0] - 1
        while self.body_dofnum[last_dof_body_id] == 0:
            last_dof_body_id -= 1
        self.qvel_end_idx = int(dof[-1] + self.body_dofnum[last_dof_body_id])

    def _set_joint(self):
        self.joint_names = list(filter(
            lambda x: self._in_scope(x), self._env.model.names[self._env.model.jnt_nameadr:]
        ))
        self.joint_ids = [
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names
        ]

        # Determine joint params
        self.jnt_qposadr = self._env.model.jnt_qposadr[self.joint_ids]
        self.jnt_type = self._env.model.jnt_type[self.joint_ids]
        self.jnt_nqpos = [self.JNT_NPOS[int(j)] for j in self.jnt_type]
        self.nq = sum(self.jnt_nqpos)
        # Determine qpos_start_idx and qpos_end_idx
        self.qpos_start_idx = int(self.jnt_qposadr[0])
        self.qpos_end_idx = int(self.jnt_qposadr[-1] + self.jnt_nqpos[-1])

    def _set_observation_space(self):
        obs = self.get_obs()
        self.obs_dim = obs.size
        low = -np.inf * np.ones(self.obs_dim)
        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _set_action_space(self):
        acts = self._xml.find('actuator')
        self.action_dim = len(list(acts))
        default = self._xml.find('default')
        range_set = False
        if default is not None:
            motor = default.find('motor')
            if motor is not None:
                ctrl = motor.get('ctrlrange')
                if ctrl:
                    clow, chigh = list(map(float, ctrl.split()))
                    high = chigh * np.ones(self.action_dim)
                    low = clow * np.ones(self.action_dim)
                    range_set = True
        if not range_set:
            high = np.ones(self.action_dim)
            low = -np.ones(self.action_dim)
        for i, motor in enumerate(list(acts)):
            ctrl = motor.get('ctrlrange')
            if ctrl:
                clow, chigh = list(map(float, ctrl.split()))
                low[i], high[i] = clow, chigh
        self._low, self._high = low, high
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_xyz(self, xyz):
        """Set (x, y, z) position of the agent; any element can be None."""
        qpos = self._env.data.qpos.ravel().copy()
        start = self.qpos_start_idx
        if xyz[0]: qpos[start] = xyz[0]
        if xyz[1]: qpos[start + 1] = xyz[1]
        if xyz[2]: qpos[start + 2] = xyz[2]
        qvel = self._env.data.qvel.ravel()
        mujoco.mj_set_state(self._env.model, self._env.data, qpos, qvel)

    # Other functions (similar to the above) ...

