import os
import numpy as np
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.utils import seeding
import mujoco
from mujoco import MjModel, MjData, viewer
import glfw


def _read_pixels(sim, width=None, height=None, camera_name=None):
    """Reads pixels w/o markers and overlay from the same camera as screen."""
    if width is None or height is None:
        resolution = glfw.get_framebuffer_size(sim.render_context_window.window)
        resolution = np.array(resolution)
        resolution = resolution * min(1000 / np.min(resolution), 1)
        resolution = resolution.astype(np.int32)
        resolution -= resolution % 16
        width, height = resolution

    img = sim.render(width, height, camera_name=camera_name, depth=False)
    img = img[::-1, :, :]  # Rendered images are upside-down.
    return img


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = MjModel.from_xml_path(fullpath)
        self.sim = MjData(self.model)
        self.data = self.sim
        self.viewer = None
        self.buffer_size = (1600, 1280)

        self.metadata = {
            'render_modes': ['human', 'rgb_array'],
            'render_fps': 60,
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if (
            type(observation) is tuple) else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds[:, 0], bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        self.sim.qpos[:] = qpos
        self.sim.qvel[:] = qvel
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.sim)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer = None
            return

        if mode == 'rgb_array':
            self.viewer_setup()
            return _read_pixels(self.sim, *self.buffer_size)
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self, mode='human'):
        if self.viewer is None and mode == 'human':
            self.viewer = viewer.launch_passive(self.model, self.sim)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        return np.concatenate([self.sim.qpos.flat, self.sim.qvel.flat])