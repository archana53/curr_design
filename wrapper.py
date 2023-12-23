# inspired from https://github.com/denisyarats/dmc2gym

import logging
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium.core import Env
from gymnasium.spaces import Box, Dict


# Create a dictionary space from the observation spec
def _spec_to_box_obs(spec, dtype=np.float32):
    def extract_min_max(s):
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        if "egocentric" in s.name:
            continue
        mn, mx = extract_min_max(s)
        if "appendages_pos" in s.name:
            mn = mn[:-3]
            mx = mx[:-3]
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape

    final_space = Dict(
        {"proprio": Box(low, high, dtype=dtype), "extero": Box(-1, 50, shape=(51,), dtype=dtype)}
    )
    return final_space


def _spec_to_box_ac(spec, dtype=np.float32):
    def extract_min_max(s):
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        if "egocentric" in s.name:
            continue
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape

    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for key, v in obs.items():
        if "egocentric" in key:
            continue
        if "appendages_pos" in key:
            v = v[:-3]
        if isinstance(v, tuple):
            v = v[0]
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMCGym(Env):
    def __init__(
        self,
        env,
        rendering="egl",
        task_kwargs={},
        render_height=64,
        render_width=64,
        render_camera_id=0,
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self._env = env

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box_obs(self._env.observation_spec().values())
        self._action_space = _spec_to_box_ac([self._env.action_spec()])
        self.sample_list = []
        # set seed if provided with task_kwargs

        self.height_field = self.init_height_field()
        self._intervals = None
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def init_height_field(self):
        x_coords = np.linspace(-1.2, 5.6, 50)
        return x_coords

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1

    # Use the modified observable from the dm_control's CMUHumanoid!
    def get_walker_pos(self, obs):
        return obs["walker/appendages_pos"][-3:]

    # Use the arena's available geoms to determine the gaps
    def set_arena_gaps(self):
        geoms = self._env._task._arena.ground_geoms
        intervals = []
        for geom in geoms:
            if geom.name and geom.name.startswith("floor"):
                pos = geom.pos
                size = geom.size
                intervals.append((pos[0], pos[0] + size[0]))
        self._intervals = intervals

    # Check whether a point in the heightfield is part of the gap or not!
    def check_gap_point(self, x):
        for i, interval in enumerate(self._intervals[:-1]):
            if interval[1] < x[0] and x[0] < self._intervals[i + 1][0]:
                return 0
        return 1

    # Create a heightfield from the walker's perspective
    def get_grid_heightfield_from_walker(self, obs):
        pos = self.get_walker_pos(obs)
        field = np.zeros((51,), dtype=np.float32)
        x_coords = self.height_field
        for i in range(50):
            field[i] = self.check_gap_point(pos + x_coords[i])
        return field, pos[2]

    # Extract extero features from the walker
    def get_extero_features(self, obs):
        field, height_from_ground = self.get_grid_heightfield_from_walker(obs)
        field[50] = height_from_ground
        return field

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
            action = action.reshape((56,))
        try:
            assert self._action_space.contains(action)
        except AssertionError:
            logging.error("Action is not in action space")
            exit
        try:
            time.sleep(0.002)
            timestep = self._env.step(action)
            obs_proprio = _flatten_obs(timestep.observation)
            obs_ext = self.get_extero_features(timestep.observation)
            observation = {"proprio": obs_proprio, "extero": obs_ext}
            reward = timestep.reward
            termination = False  # we never reach a goal
            truncation = timestep.last()
            info = {"discount": timestep.discount}
            return observation, reward, termination, truncation, info
        except Exception as e:
            logging.error("Error in step function: {:}".format(e))
            print("The action was", action)

    def reset(self, seed=None, options=None):
        if seed:
            logging.warn(
                "Currently DMC has no way of seeding episodes. It only allows to seed experiments on environment initialization"
            )

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        if self._intervals is None:
            self.set_arena_gaps()
        obs_proprio = _flatten_obs(timestep.observation)
        obs_ext = self.get_extero_features(timestep.observation)
        self._env._task._arena.mjcf_model.size.nconmax = 1000
        observation = {"proprio": obs_proprio, "extero": obs_ext}
        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

    def close(self):
        self._env.close()
