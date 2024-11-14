"""
The base class for environments based on the latest MuJoCo.
"""

import os
import sys
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco
    from mujoco import mjtFontScale, MjvCamera, MjvOption, MjvScene, MjrContext, MjrRect
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco).")


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.buffer_size = (1600, 1280)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # For the base class, we can define obs_dim based on model.nq and model.nv
        self.obs_dim = self.model.nq + self.model.nv

        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds[:, 0], bounds[:, 1]
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._seed()

        # Initialize visualization data structures
        self.camera = MjvCamera()
        self.vopt = MjvOption()
        self.scene = MjvScene(self.model, maxgeom=1000)
        self.context = MjrContext(self.model, mjtFontScale.mjFONTSCALE_150)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ------------------------------------------------------------------------

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """Called when the viewer is initialized and after every reset.
        Optionally implement this method, if you need to tinker with camera
        position and so forth.
        """
        pass

    # ------------------------------------------------------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        ob = self.reset_model()
        if ob is None:  # zihan: added, fix None observation at reset()
            ob = np.zeros(self.obs_dim)
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            width, height = self.buffer_size
            return self._read_pixels(width, height)
        elif mode == 'human':
            if self.viewer is None:
                # Note: The rendering code may need to be updated to work with the new MuJoCo viewer.
                # The following is a placeholder and may need adjustments based on your setup.
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)
                self.viewer_setup()
            # If using an interactive viewer, you might not need to call render explicitly
            # self.viewer.render()
            # For some setups, you might need to integrate with an external GUI library
            pass

    def _read_pixels(self, width, height, camera_name=None):
        """Reads pixels from the simulation."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        viewport = MjrRect(0, 0, width, height)
        # Update scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.vopt, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        # Render scene
        mujoco.mjr_render(viewport, self.scene, self.context)
        # Read pixels from framebuffer
        mujoco.mjr_readPixels(img, None, viewport, self.context)
        # Flip the image vertically
        img = img[::-1, :, :]
        return img

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
