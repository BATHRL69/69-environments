import os
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import gym

class Agent:
    """
    Superclass for all agents in the sumo MuJoCo environment.
    """

    CFRC_CLIP = 100.0

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

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, scope: str, xml_path: str, adjust_z: float = 0.0):
        self.model = model
        self.data = data
        self.scope = scope
        self.xml_path = xml_path
        self.xml = ET.parse(xml_path)
        self.adjust_z = adjust_z

        self._set_body()
        self._set_joint()

    def setup_spaces(self):
        self._set_observation_space()
        self._set_action_space()

    def _in_scope(self, name: str) -> bool:
        return name.startswith(self.scope)

    def _set_body(self):
        # Filter body names that are within the specified scope
        self.body_names = [name for name in self.model.body_names if self._in_scope(name)]
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.body_names]
        
        if -1 in self.body_ids:
            raise ValueError(f"One or more body names in scope '{self.scope}' were not found in the model.")

        self.body_name_idx = {
            name.split('/')[-1]: idx
            for name, idx in zip(self.body_names, self.body_ids)
        }

        # Determine body DOF parameters
        self.body_dofnum = np.array([self.model.dof_dofnum[i] for i in self.body_ids])
        self.body_dofadr = np.array([self.model.dof_dofadr[i] for i in self.body_ids])
        self.nv = self.body_dofnum.sum()

        # Determine qvel_start_idx and qvel_end_idx
        dof_addresses = [adr for adr in self.body_dofadr if adr >= 0]
        if not dof_addresses:
            raise ValueError("No valid DOF addresses found for bodies in scope.")
        
        self.qvel_start_idx = int(dof_addresses[0])
        last_dof_body_id = len(self.body_dofnum) - 1
        while last_dof_body_id >= 0 and self.body_dofnum[last_dof_body_id] == 0:
            last_dof_body_id -= 1
        if last_dof_body_id < 0:
            raise ValueError("All bodies have zero DOF numbers.")
        
        self.qvel_end_idx = int(dof_addresses[-1] + self.body_dofnum[last_dof_body_id])

    def _set_joint(self):
        # Filter joint names that are within the specified scope
        self.joint_names = [name for name in self.model.joint_names if self._in_scope(name)]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        
        if -1 in self.joint_ids:
            raise ValueError(f"One or more joint names in scope '{self.scope}' were not found in the model.")

        # Determine joint parameters
        self.jnt_qposadr = np.array([self.model.jnt_qposadr[i] for i in self.joint_ids])
        self.jnt_type = np.array([self.model.jnt_type[i] for i in self.joint_ids])
        self.jnt_nqpos = [self.JNT_NPOS[int(j)] for j in self.jnt_type]
        self.nq = sum(self.jnt_nqpos)

        # Determine qpos_start_idx and qpos_end_idx
        self.qpos_start_idx = int(self.jnt_qposadr[0])
        self.qpos_end_idx = int(self.jnt_qposadr[-1] + self.jnt_nqpos[-1])

    def _set_observation_space(self):
        obs = self.get_obs()
        self.obs_dim = obs.size
        low = -np.inf * np.ones(self.obs_dim, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _set_action_space(self):
        acts = self.xml.find('actuator')
        if acts is None:
            raise ValueError("No 'actuator' tag found in XML.")
        
        act_list = list(acts)
        self.action_dim = len(act_list)
        default = self.xml.find('default')
        range_set = False

        if default is not None:
            motor = default.find('motor')
            if motor is not None:
                ctrl = motor.get('ctrlrange')
                if ctrl:
                    clow, chigh = map(float, ctrl.split())
                    high = chigh * np.ones(self.action_dim, dtype=np.float32)
                    low = clow * np.ones(self.action_dim, dtype=np.float32)
                    range_set = True

        if not range_set:
            high = np.ones(self.action_dim, dtype=np.float32)
            low = -np.ones(self.action_dim, dtype=np.float32)

        for i, motor in enumerate(act_list):
            ctrl = motor.get('ctrlrange')
            if ctrl:
                clow, chigh = map(float, ctrl.split())
                low[i], high[i] = clow, chigh

        self._low, self._high = low, high
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_xyz(self, xyz: tuple):
        """Set (x, y, z) position of the agent; any element can be None."""
        qpos = self.data.qpos.copy()
        start = self.qpos_start_idx
        if xyz[0] is not None:
            qpos[start] = xyz[0]
        if xyz[1] is not None:
            qpos[start + 1] = xyz[1]
        if xyz[2] is not None:
            qpos[start + 2] = xyz[2]
        qvel = self.data.qvel.copy()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def set_euler(self, euler: tuple):
        """Set Euler angles of the agent; any element can be None."""
        qpos = self.data.qpos.copy()
        start = self.qpos_start_idx
        if euler[0] is not None:
            qpos[start + 4] = euler[0]
        if euler[1] is not None:
            qpos[start + 5] = euler[1]
        if euler[2] is not None:
            qpos[start + 6] = euler[2]
        qvel = self.data.qvel.copy()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def set_opponents(self, opponents: list):
        self._opponents = opponents

    def reset(self):
        # Implement reset logic if necessary
        pass

    # --------------------------------------------------------------------------
    # Various Getters
    # --------------------------------------------------------------------------

    def get_body_com(self, body_name: str) -> np.ndarray:
        full_name = f"{self.scope}/{body_name}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_name)
        if body_id == -1:
            raise ValueError(f"Body name '{full_name}' not found.")
        # data.subtree_com is a flat array: [x0, y0, z0, x1, y1, z1, ...]
        com = self.data.subtree_com[body_id * 3:(body_id + 1) * 3].copy()
        return com

    def get_cfrc_ext(self, body_ids: list = None) -> np.ndarray:
        if body_ids is None:
            body_ids = self.body_ids
        # data.cfrc_ext is a flat array: [fx0, fy0, fz0, mx0, my0, mz0, ...]
        cfrc_ext = np.array([
            self.data.cfrc_ext[bid * 6 : (bid + 1) * 6]
            for bid in body_ids
        ]).flatten()
        return cfrc_ext

    def get_qpos(self) -> np.ndarray:
        """Retrieve and adjust the agent's qpos."""
        qpos = self.data.qpos[self.qpos_start_idx:self.qpos_end_idx].copy()
        qpos[2] += self.adjust_z
        return qpos

    def get_qvel(self) -> np.ndarray:
        """Retrieve the agent's qvel."""
        qvel = self.data.qvel[self.qvel_start_idx:self.qvel_end_idx].copy()
        return qvel

    def get_qfrc_actuator(self) -> np.ndarray:
        """Retrieve the agent's actuator forces."""
        qfrc = self.data.qfrc_actuator[self.qvel_start_idx:self.qvel_end_idx].copy()
        return qfrc

    def get_cvel(self) -> np.ndarray:
        """Retrieve the agent's centroidal velocities."""
        # data.cvel is a flat array: [cvel0_x, cvel0_y, cvel0_z, ...]
        cvel = np.array([
            self.data.cvel[bid * 6 : (bid + 1) * 6]
            for bid in self.body_ids
        ]).flatten()
        return cvel

    def get_body_mass(self) -> np.ndarray:
        """Retrieve the masses of the agent's bodies."""
        body_mass = self.model.body_mass[self.body_ids].copy()
        return body_mass

    def get_xipos(self) -> np.ndarray:
        """Retrieve the inertial positions of the agent's bodies."""
        # data.xipos is a flat array: [x0, y0, z0, x1, y1, z1, ...]
        xipos = np.array([
            self.data.xipos[bid * 3 : (bid + 1) * 3]
            for bid in self.body_ids
        ]).flatten()
        return xipos

    def get_cinert(self) -> np.ndarray:
        """Retrieve the inertial properties of the agent's bodies."""
        # model.cinert is a flat array, 10 values per body
        cinert = np.array([
            self.model.cinert[bid * 10 : (bid + 1) * 10]
            for bid in self.body_ids
        ]).flatten()
        return cinert

    def get_obs(self) -> np.ndarray:
        """Construct the observation vector for the agent."""
        # Observe self
        self_forces = np.abs(np.clip(
            self.get_cfrc_ext(), -self.CFRC_CLIP, self.CFRC_CLIP))
        obs = [
            self.get_qpos(),          # Self positions
            self.get_qvel(),          # Self velocities
            self_forces,              # Self forces
        ]

        # Observe opponents
        for opp in self._opponents:
            body_ids = [
                opp.body_name_idx[name]
                for name in ['torso']
                if name in opp.body_name_idx
            ]
            opp_forces = np.abs(np.clip(
                opp.get_cfrc_ext(body_ids), -self.CFRC_CLIP, self.CFRC_CLIP))
            obs.extend([
                opp.get_qpos()[:7],      # Opponent torso position
                opp_forces,              # Opponent torso forces
            ])
        
        return np.concatenate(obs)

    def before_step(self):
        """Store the position before taking a simulation step."""
        self.posbefore = self.get_qpos()[:2].copy()

    def after_step(self, action: np.ndarray) -> float:
        """Process after taking a simulation step and compute the reward."""
        self.posafter = self.get_qpos()[:2].copy()
        # Control cost
        reward = - self.COST_COEFS['ctrl'] * np.square(action).sum()
        return reward


# ------------------------------------------------------------------------------
# Specific Agent Implementations
# ------------------------------------------------------------------------------

class Ant(Agent):
    """The 4-leg agent."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, **kwargs):
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "ant.xml")
        super(Ant, self).__init__(model, data, scope="ant", xml_path=xml_path, **kwargs)


class Bug(Agent):
    """The 6-leg agent."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, **kwargs):
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "bug.xml")
        super(Bug, self).__init__(model, data, scope="bug", xml_path=xml_path, **kwargs)


class Spider(Agent):
    """The 8-leg agent."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, **kwargs):
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "spider.xml")
        super(Spider, self).__init__(model, data, scope="spider", xml_path=xml_path, **kwargs)


# ------------------------------------------------------------------------------
# Agent Factory
# ------------------------------------------------------------------------------

_available_agents = {
    'ant': Ant,
    'bug': Bug,
    'spider': Spider,
}

def get_agent(name: str, model: mujoco.MjModel, data: mujoco.MjData, **kwargs) -> Agent:
    """
    Factory function to retrieve an agent instance by name.

    Args:
        name (str): Name of the agent ('ant', 'bug', 'spider').
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
        **kwargs: Additional keyword arguments.

    Returns:
        Agent: An instance of the specified agent.

    Raises:
        ValueError: If the agent name is not available.
    """
    if name not in _available_agents:
        raise ValueError(f"Agent '{name}' is not available. Choose from {list(_available_agents.keys())}.")
    return _available_agents[name](model, data, **kwargs)
